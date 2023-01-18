# Development Progress Tracker
Legend: Y=Yes, No=Not Planned, P=Pending

## version 0.1.5
- 0.1.5 will be the last version for major developments in TF1
- next release will be 0.2.0 and pytorch dev will be the focus

| module    |                            | TF1       | Torch | TF2   |
| --------- | -------------------------- | --------- | ----- | ----- |
| architect | GeneralController          | Y         | Y     | Y     |
| architect | ProbaModelBuildGeneticAlgo | Y         | Y     | Y     |
| architect | AmbientController          | Y         | N     | N     |
| architect | MultiIOController          | Y         | N     | N     |
| modeler   | resnet                     | Y         | Y     | Y     |
| modeler   | supernet                   | Y         | Y     | N     |
| modeler   | sequential                 | Y         | Y     | Y     |
| modeler   | gnn                        | N         | **P** | N     |
| modeler   | kinn                       | P (paper) | N     | P     |
| modeler   | sparse_ffnn                | Y         | N     | N     |
| --------- | -------------------------- | -----     | ----- | ----- |
| Total     |                            | 8(1)      | 5(1)  | 4(1)  |
| Test      |                            | 90%       | n/a   | n/a   |


