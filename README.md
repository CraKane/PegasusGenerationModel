# PegasusGenerationModel
## Introduction
- It's a project for building a deep learning model generating Pagesus.
- It only uses horses and birds dataset from CIFAR-10 or STL-10.
## Method
- Building a VAE model with transfer learning.
- Training the model by the dataset we processed.
![Horses examples I selected](horse.png)
\\
![Birds examples I selected](bird.png)
- 2-channels encoder
![Model](model.png)
- As for horses and birds dataset, we choosed the figures that birds' wings and horses' head can be seen clearly.
![Results](pegasus.png)
## Code folder
- "processed" is the dataset we processed by dataProcess.py
- The detailed code is in pegasus-code.ipynb
## Notes
- Other details are in the report with ICLR 2021 template.
