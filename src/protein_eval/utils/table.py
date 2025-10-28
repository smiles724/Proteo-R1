from prettytable import PrettyTable, HRuleStyle


def dict_to_table(data_dict, headers=None):
    """
    将字典(键为字符串，值为列表)转换为prettytable表格，
    每个键值对对应表格的一行

    参数:
        data_dict: 字典，键为字符串，值为列表
        headers: 表头列表，如果为None，则自动生成

    返回:
        可打印的PrettyTable对象
    """
    # 创建PrettyTable对象
    table = PrettyTable()
    table.hrules = HRuleStyle.ALL

    # 确定所有列表中的最大长度
    max_list_length = max(len(value) for value in data_dict.values())

    # 设置列名
    if headers is None:
        # 默认第一列表头为"名称"，其他列为"值1"，"值2"...
        headers = ["名称"] + [f"值{i + 1}" for i in range(max_list_length)]

    table.field_names = headers

    # 添加行数据
    for key, value_list in data_dict.items():
        # 创建行：第一个元素是键，后面是值列表的所有元素
        row = [key] + value_list + [""] * (max_list_length - len(value_list))
        table.add_row(row)

    return table


def list_to_table(data_list, headers=None):
    # 创建PrettyTable对象
    table = PrettyTable()
    table.hrules = HRuleStyle.ALL

    # 确定所有列表中的最大长度
    max_list_length = max(len(value) for value in data_list)

    # 设置列名
    if headers is None:
        headers = [f"值{i + 1}" for i in range(max_list_length)]

    table.field_names = headers

    # 添加行数据
    for value_list in data_list:
        row = value_list + [""] * (max_list_length - len(value_list))
        table.add_row(row)

    return table

