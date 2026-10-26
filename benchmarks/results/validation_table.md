| Benchmark | Quantity | FlowTrace | Reference | Acceptance | Result |
|---|---|---|---|---|---|
| `taylor_green_order` | observed order (finest pair) | 1.998  | 2 | > 1.8 | pass |
| `mms_mac` | observed order u (L2) | 2.006  | 2 | > 1.8 | pass |
| `mms_mac` | observed order v (L2) | 2.006  | 2 | > 1.8 | pass |
| `mms_mac` | observed order p (L2) | 1.998  | 2 | > 1.8 | pass |
| `mms_mac` | observed order T (L2) | 2  | 2 | > 1.8 | pass |
| `mms_simplec_quads` | observed order u (L2) | 2.009  | 2 | > 1.8 | pass |
| `mms_simplec_quads` | observed order v (L2) | 2.011  | 2 | > 1.8 | pass |
| `mms_simplec_quads` | observed order p (L2) | 1.991  | 2 | > 1.8 | pass |
| `mms_simplec_quads` | observed order T (L2) | 1.977  | 2 | > 1.8 | pass |
| `mms_simplec_tri` | observed order u (L2) | 2.028  | 2 | > 1.8 | pass |
| `mms_simplec_tri` | observed order v (L2) | 2.021  | 2 | > 1.8 | pass |
| `mms_simplec_tri` | observed order p (L2) | 1.199  | 2 | > 1.0 | pass |
| `mms_simplec_tri` | observed order T (L2) | 1.995  | 2 | > 1.8 | pass |
| `projection_divergence` | max|div u| fast separable solver | 6.939e-15 1/s | 0 | < 1e-10 | pass |
| `projection_divergence` | max|div u| multigrid-PCG (obstacle) | 3.400e-12 1/s | 0 | < 1e-8 | pass |
| `cavity_re100_mac` | max |du| centreline | 0.004195  | 0 | < 0.01 | pass |
| `cavity_re100_mac` | max |dv| centreline | 0.008457  | 0 | < 0.015 | pass |
| `cavity_re100_simplec_quads` | max |du| centreline | 0.004236  | 0 | < 0.01 | pass |
| `cavity_re100_simplec_quads` | max |dv| centreline | 0.008108  | 0 | < 0.015 | pass |
| `cavity_re100_simplec_tri` | max |du| centreline | 0.005587  | 0 | < 0.015 | pass |
| `cavity_re100_simplec_tri` | max |dv| centreline | 0.008918  | 0 | < 0.02 | pass |
| `cavity_re1000_simplec` | max |du| centreline | 0.005342  | 0 | < 0.02 | pass |
| `cavity_re1000_simplec` | max |dv| centreline | 0.009978  | 0 | < 0.02 | pass |
| `natural_convection_ra1e4_mac` | Nu (hot wall) | 2.251  | 2.243 | within 1.5 % | pass |
| `natural_convection_ra1e4_mac` | u_max on x=0.5 | 16.19  | 16.18 | within 2 % | pass |
| `natural_convection_ra1e4_mac` | v_max on y=0.5 | 19.64  | 19.62 | within 2 % | pass |
| `natural_convection_ra1e5_fmg` | Nu (hot wall) | 4.569  | 4.519 | within 2 % | pass |
| `natural_convection_ra1e3_simplec` | Nu (hot wall) | 1.118  | 1.118 | within 1.5 % | pass |
| `natural_convection_ra1e3_simplec` | energy imbalance |sum Q| / Q_hot | 2.503e-13  | 0 | < 1e-4 | pass |
| `channel_re50_mac` | outlet profile max |u - u_Poiseuille| | 0.005777  | 0 | < 0.01 | pass |
| `channel_re50_mac` | dp/dx (developed region) | -0.2383  | -0.24 | within 2 % | pass |
| `simplec_mass_balance` | net mass imbalance / inflow, channel | 8.298e-11  | 0 | < 1e-8 | pass |
| `simplec_mass_balance` | net mass imbalance / inflow, step mesh | 4.479e-10  | 0 | < 1e-8 | pass |
| `gci_natural_convection` | observed order p | 2.137  | 2 | 1.5 < p < 2.5 | pass |
| `gci_natural_convection` | Richardson-extrapolated Nu | 2.245  | 2.243 | within 1 % | pass |
| `gci_natural_convection` | asymptotic-range ratio | 1  | 1 | 0.95-1.05 | pass |
| `gci_cavity` | observed order p | 2.106  | 2 | 1.5 < p < 2.5 | pass |
| `channel_retau395_sst` | k-omega-sst: max |U+ - DNS| (y+ >= 1) | 4.067 % | 0 | < 5.0 % | pass |
| `channel_retau395_sst` | k-omega-sst: wall shear (exact 1 at equilibrium) | 0.9994  | 1 | within 1 % | pass |
| `channel_retau395_kepsilon` | k-epsilon-standard: max |U+ - DNS| (y+ >= 51) | 3.525 % | 0 | < 5.0 % | pass |
| `channel_retau395_kepsilon` | k-epsilon-standard: wall shear (exact 1 at equilibrium) | 0.9999  | 1 | within 1 % | pass |
| `channel_retau395_kepsilon` | k-epsilon-realizable: max |U+ - DNS| (y+ >= 51) | 7.575 % | 0 | < 10.0 % | pass |
| `channel_retau395_kepsilon` | k-epsilon-realizable: wall shear (exact 1 at equilibrium) | 0.9999  | 1 | within 1 % | pass |
| `flat_plate_sst` | mean deviation from White, 1e6 < Re_x < 1e7 | -9.844 % | 0 | within 12 % | pass |
| `flat_plate_sst` | mean deviation from Schultz-Grunow, same range | -5.317 % | 0 | within 6 % | pass |
| `flat_plate_kepsilon` | mean deviation from White, 1e6 < Re_x < 1e7 | -11.41 % | 0 | within 12 % | pass |
| `flat_plate_kepsilon` | mean deviation from Schultz-Grunow, same range | -6.969 % | 0 | within 6 % | **FAIL** |
| `bfs_driver_seegmiller_sst` | reattachment length x_r / H | 6.608  | 6.26 | within 10 % | pass |
| `taylor_couette` | rotating wall: max |u_theta - exact|, 16x96 | 0.002739  | 0 | < 5e-3 | pass |
| `taylor_couette` | rotating wall: observed order | 2.014  | 2 | > 1.8 | pass |
| `taylor_couette` | MRF inner ring: max |u_theta - exact|, 16x96 | 0.002636  | 0 | < 5e-3 | pass |
| `taylor_couette` | MRF inner ring: observed order | 1.999  | 2 | > 1.8 | pass |
| `periodic_poiseuille` | quads 4x16: max |u - exact| / u_max | 0.003906  | 0 | < 0.01 | pass |
| `periodic_poiseuille` | triangles 4x16: max |u - exact| / u_max | 0.00204  | 0 | < 0.01 | pass |
| `periodic_poiseuille` | observed order (quads) | 2  | 2 | > 1.8 | pass |
| `cht_composite_wall` | max |T - exact| | 2.092e-13  | 0 | < 1e-10 | pass |
| `cht_composite_wall` | heat flux | 0.3125 W/m | 0.3125 | within 1e-10 | pass |
| `cht_heated_channel` | heat crossing the fluid-solid interface | 4 W/m | 4 | within 1e-6 | pass |
| `cht_heated_channel` | global energy imbalance | 9.132e-13 W/m | 0 | < 1e-8 | pass |
| `cht_heated_channel` | enthalpy rise inlet -> outlet | 4 W/m | 4 | within 1e-6 | pass |
| `warmstart_proxy` | cavity: iterations to the converged field, cold / warm | 2.381 x | - | > 1 | pass |
| `warmstart_proxy` | cavity: warm run reaches the cold-start solution | 9.943e-05  | 0 | <= 0.0001 | pass |
| `warmstart_proxy` | channel: iterations to the converged field, cold / warm | 1.447 x | - | > 1 | pass |
| `warmstart_proxy` | channel: warm run reaches the cold-start solution | 9.477e-05  | 0 | <= 0.0001 | pass |
| `warmstart_proxy` | step: iterations to the converged field, cold / warm | 1.847 x | - | > 1 | pass |
| `warmstart_proxy` | step: warm run reaches the cold-start solution | 9.985e-05  | 0 | <= 0.0001 | pass |
| `warmstart_proxy` | cylinder: iterations to the converged field, cold / warm | 0.6452 x | - | > 1 | **FAIL** |
| `warmstart_proxy` | cylinder: warm run reaches the cold-start solution | 9.802e-05  | 0 | <= 0.0001 | pass |
| `warmstart_proxy` | heated: iterations to the converged field, cold / warm | 2 x | - | > 1 | pass |
| `warmstart_proxy` | heated: warm run reaches the cold-start solution | 9.947e-05  | 0 | <= 0.0001 | pass |

_Measured 2026-10-26T06:23:39 on AMD Ryzen 9 7940HS w/ Radeon 780M Graphics, GPU NVIDIA GeForce RTX 4060 Laptop GPU; tier `full`._
