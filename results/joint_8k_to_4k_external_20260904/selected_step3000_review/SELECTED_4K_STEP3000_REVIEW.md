# Selected 4K Joint Step3000 Review

## Endpoint ladder and scalability

| Dataset | Sequence | 4K Base between 2K/8K | 4K Full between 2K/8K | Δbpp Full−Base | ΔdB Full−Base | Scalable |
|---|---|---:|---:|---:|---:|---:|
| 8iVFB | Longdress 1300 | FAIL | PASS | 0.144658 | +1.3272 | PASS |
| 8iVFB | Loot 1200 | FAIL | PASS | 0.001868 | -0.0147 | WARNING |
| 8iVFB | Redandblack 1550 | FAIL | PASS | 0.046721 | +0.3837 | PASS |
| 8iVFB | Soldier 0690 | FAIL | PASS | 0.006655 | +0.0464 | PASS |
| Owlii | Basketball 0200 | FAIL | PASS | 0.010731 | +0.2900 | PASS |
| Owlii | Dancer 0001 | FAIL | PASS | 0.013139 | +0.2698 | PASS |
| Owlii | Exercise 0001 | FAIL | PASS | 0.007936 | +0.1594 | PASS |
| Owlii | Model 0001 | FAIL | PASS | 0.036573 | +0.7448 | PASS |

Scalable legality requires exact hard decoding, four Base residual streams, no native r5, identical Base prefix bits in Base/Full, and `Full_bits = Base_bits + Enhancement_bits`. A positive quality increment is reported separately as RD usefulness.
