# DJCM

This repo is the Pytorch implementation of ["DJCM: A Deep Joint Cascade Model for Singing Voice Separation and Vocal Pitch Estimation"](https://arxiv.org/abs/2401.03856). 

Model Link: [DJCM Model](https://huggingface.co/AnhP/DJCM-Test)

Current quality:

| index |   sdr±std   | gnsdr |   rpa±std   |   rca±std   |    oa±std    |
|-------|-------------|-------|-------------|-------------|--------------|
| 11776	| NONE        | NONE  | 88.87±7.25  | 89.9±6.74	  | 34.2±22.21   |
| 14720	| NONE        | NONE  | 89.09±7.47  | 90.15±6.88  | 39.83±24.26  |
| 17664	| NONE        | NONE  | 89.47±7.16  | 90.44±6.65  | 37.56±25.77  |
| 20608	| NONE        | NONE  | 90.49±6.63  | 91.29±6.24  | 37.71±25.67  |
| 22080	| NONE        | NONE  | 90.65±6.69  | 91.56±6.15  | 41.01±24.69  |