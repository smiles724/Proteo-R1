import logging
import os
import sys
import ray
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput
import pickle
from typing import Any, Callable, Optional
import zmq
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm import SamplingParams
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.inputs import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM
from verl.utils.fs import copy_to_local
from verl.workers.rollout.async_server import AsyncServerBase
from vllm.entrypoints.logger import RequestLogger

logger = logging.getLogger(__name__)

@ray.remote(num_cpus=1)
class AsyncvLLMServer(AsyncServerBase):
    """
    AsyncvLLMServer is a wrapper for AsyncLLM.
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str):
        super().__init__()
        self.config = config.actor_rollout_ref
        self.vllm_dp_rank = vllm_dp_rank
        self.engine: AsyncLLM = None
        self.generate_call_count = 0

    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        print(f"[vLLM Async Server] Starting engine initialization")
        
        rollout_config = self.config.rollout
        
        model_path = self.config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        print(f"[vLLM Async Server] Model path: {model_path}")
        
        local_path = copy_to_local(model_path)
        print(f"[vLLM Async Server] Local path: {local_path}")
        
        trust_remote_code = self.config.model.get("trust_remote_code", False)
        
        external_lib = self.config.model.get("external_lib", None)
        if external_lib:
            print(f"[vLLM Async Server] Importing external library (in parent process): {external_lib}")
            from verl.utils.import_utils import import_external_libs
            import_external_libs(external_lib)
            print(f"[vLLM Async Server] Successfully imported external library.")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.vllm_dp_rank)
        
        tensor_parallel_size = rollout_config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = rollout_config.get("max_num_batched_tokens", 8192)
        max_model_len = rollout_config.max_model_len if rollout_config.max_model_len else rollout_config.prompt_length + rollout_config.response_length
        self.max_model_len = int(max_model_len)

        kwargs = dict(n=1, logprobs=0, repetition_penalty=1.0, max_new_tokens=rollout_config.response_length)
        for k in rollout_config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = rollout_config.get(k)

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=rollout_config.free_cache_engine,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            dtype=rollout_config.dtype,
            enforce_eager=rollout_config.enforce_eager,
            gpu_memory_utilization=rollout_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            max_model_len=self.max_model_len,
            load_format="auto",
            disable_log_stats=rollout_config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=rollout_config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=rollout_config.get("seed", 0),
        )
        
        print(f"[vLLM Async Server] Intended TP={tensor_parallel_size}, DP=1, PP=1")
        vllm_config = engine_args.create_engine_config()
        print(f"[vLLM Async Server] Engine config created successfully")

        self.engine = AsyncLLM.from_vllm_config(vllm_config)
        print(f"[vLLM Async Server] AsyncLLM engine created successfully")

        model_config = self.engine.model_config
        BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
        
        self.openai_serving_chat = OpenAIServingChat(
            self.engine, model_config, models, "assistant",
            request_logger=RequestLogger(max_log_len=4096),
            chat_template=None, chat_template_content_format="auto",
            enable_auto_tools=rollout_config.multi_turn.tool_config_path is not None,
            tool_parser=rollout_config.multi_turn.format,
        )

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint. This method is required by the base class."""
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str) -> list[int]:
        self.generate_call_count += 1
        if self.generate_call_count <= 10:
            print(f"[vLLM Async Server] Generate call #{self.generate_call_count} - request_id: {request_id}, prompt_len: {len(prompt_ids)}")
        
        max_tokens = self.max_model_len - len(prompt_ids)
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        generator = self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)
        
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        if self.generate_call_count <= 10:
            print(f"[vLLM Async Server] Generate call #{self.generate_call_count} completed - output_len: {len(final_res.outputs[0].token_ids)}")
        
        return final_res.outputs[0].token_ids

    async def wake_up(self):
        if self.config.rollout.free_cache_engine:
            await self.engine.wake_up()

    async def sleep(self):
        await self.engine.reset_prefix_cache()
        if self.config.rollout.free_cache_engine:
            await self.engine.sleep()
