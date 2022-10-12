| Notebook:     | forum_pytorch | yc930 | pytorch_tutorial |
| ----------- | ----------- | ----------- | ----------- |
| Learning rate      | 3e-5       | 1e-4 | 1e-2
| Target net update    | 200 (steps)       | Na | 10 (episodes)
| Batch size   | 256        | 128       | 128
| Gamma   | 1        | 0.999 | 0.999
| Memory size   | 10 000        | 10 000 | 10 000
| Memory alpha   | 0.6        | NA | NA
| Memory beta start   | 0.4       | NA | NA
| Memory beta frames   | 10 000       |NA | NA
| eps start  | 1.0       | 0.4 | 0.9
| eps end   | 0.01       | 0.05 | 0.05
| eps decay   | 10      | 200   | 200
| optimizer  | Adam     | Adam (weight_decay=1e-5)   | RMSprop
| loss   | MSE      | Huber   | Huber
| preprocessing (state) | stack last 3 frames (greyscale)     | difference between previous and current | difference between previous and current | 