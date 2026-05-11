# ProteoR1 Demo Data

This directory contains small public demo entries for CPU-side smoke tests and
quickstart documentation. Each entry includes one CIF file, a `design_points.txt`
file, and a short entry README.

## Entries

| Entry | Type | Demo design points |
| --- | --- | --- |
| `8q7o_A__C` | nano antibody | `[C,4], [C,1], [C,71]` |
| `8sxp_H_L_C` | paired antibody | `[C,68], [C,34], [C,32]` |
| `8slb_H_L_A` | paired antibody | `[A,62], [A,70], [A,60]` |
| `8r9y_H_L_A` | paired antibody | `[A,125], [A,61], [A,126]` |
| `8tg9_H_L_A` | paired antibody | `[A,295], [A,288], [A,292]` |

## Layout

Each entry has this shape:

```text
data/demo/<entry>/
  <entry>.cif
  design_points.txt
```

`design_points.txt` uses the antigen hotspot syntax accepted by
`proteor1-prepare-cdr`, for example `[C,4], [C,1], [C,71]`. These are
antigen-chain positions that condition the understand model prompt; CDR masking
is inferred from the antibody chains in the entry id.
