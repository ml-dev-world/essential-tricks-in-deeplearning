# Essential Tricks In Deep Learning

<p align="center">
  <img src="https://assets.spe.org/dims4/default/7c6d2d6/2147483647/strip/true/crop/1051x552+0+0/resize/1200x630!/quality/90/?url=http%3A%2F%2Fspe-brightspot.s3.amazonaws.com%2F53%2F9d%2F228eca9b412bb1e3aa8b76d5f9db%2Fdaaiml.jpg" alt="Deep Learning">
</p>

Master key techniques to enhance your deep learning skills. This repository is a concise guide to essential tricks for optimizing model performance.

## Table of Contents

### Introduction to Neural Networks and Model Building:
1. [Building Your First Neural Network](./01_Introduction_to_Neural_Networks_and_Model_Building/01_Building_Your_First_Neural_Network.ipynb) - Introduction to neural networks for beginners.
2. [Pixels To Prediction Using ConvNet](./01_Introduction_to_Neural_Networks_and_Model_Building/02_Pixels_To_Prediction_Using_ConvNet.ipynb) - Building and training your first Convolutional Neural Network.

### Data Loading and Processing
1. [Memory Efficient DataLoader And Parallelization](./02_Data_Loading_and_Processing/07_Memory_Efficient_DataLoader_And_Parallelization.ipynb) - Optimize data loading and processing for memory efficiency.
2. [Data Augmentation Using torchvision](./02_Data_Loading_and_Processing/08_Data_Augmentation_Using_torchvision.ipynb) - Implement data augmentation techniques using the torchvision library.

### Model Initialization and Regularization:
1. [Decoding Dropout Technique](./03_Model_Initialization_and_Regularization/04_Decoding_Dropout_Technique.ipynb) - Unravel the dropout technique and its impact on model training.
2. [Optimizing Convolutions With Batchnorm](./03_Model_Initialization_and_Regularization/05_Optimizing_Convolutions_With_Batchnorm.ipynb) - Enhance model efficiency using batch normalization with convolutions.
3. [Improving Generalization With Label Smoothing](./03_Model_Initialization_and_Regularization/09_Improving_Generalization_With_Label_Smoothing.ipynb) - Enhance model generalization using label smoothing.
4. [Exploring Model Initialization Strategies](./03_Model_Initialization_and_Regularization/06_Exploring_Model_Initialization_Strategies.ipynb) - Dive into various strategies for initializing neural network weights.

### Training Schedule and Strategies:
1. [Dynamic LR Scheduling](./04_Training_Schedule_and_Strategies/11_Dynamic_LR_Scheduling.ipynb) - Implement dynamic learning rate scheduling techniques.
2. [Optimizing Learning Rate With LR Finder](./04_Training_Schedule_and_Strategies/12_Optimizing_Learning_Rate_With_LR_Finder.ipynb) - Optimize learning rates using a learning rate finder.
3. [Warmup Strategies With Cosine Annealing](./04_Training_Schedule_and_Strategies/13_Warmup_Strategies_With_Cosine_Annealing.ipynb) - Implement warmup strategies with cosine annealing for better convergence.
4. [Early Stopping Strategies For Training](./04_Training_Schedule_and_Strategies/20_Early_Stopping_Strategies_For_Training.ipynb) - Selection of a reasonably optimal model.
    
### Gradient Optimization
1. [Adaptive Gradient Clipping](./05_Gradient_Optimization/14_Adaptive_Gradient_Clipping.ipynb) - Explore gradient clipping and adaptive gradient clipping techniques.
2. [Smoothing Gradients With Gradient Penalty](./05_Gradient_Optimization/16_Smoothing_Gradients_With_Gradient_Penalty.ipynb) - Implement gradient penalty for smoother training gradients.
3. [Accumulating Gradient For Efficient Training](./05_Gradient_Optimization/17_Accumulating_Gradient_For_Efficient_Training.ipynb) - Optimize training efficiency by accumulating gradients.
4. [Controlling Overfitting With Weight Decay](./05_Gradient_Optimization/15_Controlling_Overfitting_With_Weight_Decay.ipynb) - Mitigate overfitting using weight decay and the AdamW optimizer.
5. [Memory Efficient Models With Checkpointing](./05_Gradient_Optimization/29_Memory_Efficient_Models_with_Checkpointing.ipynb) - Efficient memory usage in training.

### Precision and Efficiency
1. [Automatic Mixed Precision Training](./06_Precision_and_Efficiency/18_Automatic_Mixed_Precision_Training.ipynb) -  Accelerate training by using a combination of lower-precision and higher-precision numerical representations.

## Experiment Management and Monitoring
1. [Dynamic Progress Bar Using tqdm](./07_Experiment_Management_and_Monitoring/03_Dynamic_Progress_Bar_Using_tqdm.ipynb) - Explore model initialization strategies and minor bug fixes.
2. [Ensuring Experiment Reproducibility](./07_Experiment_Management_and_Monitoring/10_Ensuring_Experiment_Reproducibility.ipynb) - Implement practices to ensure reproducibility in your experiments.
3. [Effective Model Checkpointing](./07_Experiment_Management_and_Monitoring/19_Effective_Model_Checkpointing.ipynb) - Selecting the best-performing model.
4. [Experiment Tracking With mlflow](./07_Experiment_Management_and_Monitoring/21_Experiment_Tracking_With_mlflow.ipynb) - mlflow for Experiment Tracking
5. [Logging Model Parameters](./07_Experiment_Management_and_Monitoring/22_Logging_Model_Parameters.ipynb) - Logging model parameters - flops, trainable params, size etc.
6. [Understanding CUDA Memory Usage](./07_Experiment_Management_and_Monitoring/23_Understanding_CUDA_Memory_Usage.ipynb) - Manage GPU Usage.

### Hyperparameter Optimization
1. [Hyperparameter Tuning With optuna](./08_Hyperparameter_Optimization/24_Hyperparameter_Tuning_With_optuna.ipynb) - Automate the optimization process of hyperparameters.
2. [Hyperparameter Tuning With optuna - Training](./08_Hyperparameter_Optimization/24_2_Hyperparameter_Tuning_With_optuna.ipynb) - Notebook demonstrating training with optimal hyperparams.

### Model Evaluation and Analysis
1. [Deep Dive Into Error Analysis](./09_Model_Evaluation_and_Analysis/25_Deep_Dive_Into_Error_Analysis.ipynb) - How to calculate and evaluate your model.
2. [Understanding Confusion Matrix in Deep Learning](./09_Model_Evaluation_and_Analysis/26_Understanding_Confusion_Matrix_In_Deep_Learning.ipynb) - Implement confusion matrix.
3. [Classwise Metrics for Model Evaluation](./09_Model_Evaluation_and_Analysis/27_Classwise_Metrics_For_Model_Evaluation.ipynb) - How to calculate class wise metrics.
4. [Model Interpretibilit With captum](./09_Model_Evaluation_and_Analysis/31_Model_Interpretibility_With_captum.ipynb) - Exploring Model Insights with Interpretability.

### Performance Optimization
1. [`torch.compile`](./10_Performance_Optimization/39_Torch_Compile.ipynb) - latest method to speed up your PyTorch code! Run PyTorch code faster by JIT-compiling PyTorch code into optimized kernels, all while requiring minimal code changes.
2. [Model Serialization and Export](./10_Performance_Optimization/40_Model_Serialization_&_Export.ipynb) - Multiple model export strategies like state dict, ONNX, and TorchScript. How to convert PyTorch models to TensorFlow and TensorFlow Lite, with guidance on inference execution.

### Transfer Learning
1. [Going Deeper with Transfer Learning](./11_Transfer_Learning/32_Going_Deeper_With_Transfer_Learning.ipynb) - Leverage knowledge gained from solving one problem to improve performance on another problem.
2. [Freeze Unfreeze Backbone](./11_Transfer_Learning/33_Freezing_Backbone.ipynb) - Selectively enabling or disabling the training of specific layers
3. [Differential Learning Rate](./11_Transfer_Learning/34_Differential_Learning_Rate.ipynb) - Different learning rates are applied to different parameters.
4. [Layerwise LR Decay](./11_Transfer_Learning/35_Layerwise_Learning_Rate_Decay.ipynb) -  Training deep neural networks to adjust the learning rate for each layer individually.

### Advanced Training Techniques
1. [Stochastic Weight Averaging For Improved Convergence](./12_Advanced_Training_Techniques/28_Stochastic_Weight_Averaging_For_Improved_Convergence.ipynb) - Ensemble model via weight averaging.
2. [Improving Stability With EMA](./12_Advanced_Training_Techniques/36_Improving_Stability_With_EMA.ipynb) -  Stabilize training and improve model performance.
3. [Progressive Resizing](./12_Advanced_Training_Techniques/37_Progressive_Resizing.ipynb) - Adjust the size of images progressively based on the epoch.
4. [Online Hard Negative Mining](./12_Advanced_Training_Techniques/38_Online_Hard_Negative_Mining.ipynb) - Prioritize hard examples during training.

### Testing
1. [Enhancing Testing with Test Time Augmentation](./13_Testing/30_Enhancing_Testing_With_Test_Time_Augmentation.ipynb) - Enhancing model predictions with augmentation.

## YouTube Playlist

<p align="center">
  <a href="https://www.youtube.com/playlist?list=PL4HNImpE6EWinFM0YutqEAigEFhcYtmtX">
    <img src="https://i.ytimg.com/vi/LvP-hmWGex4/hqdefault.jpg?sqp=-oaymwEXCNACELwBSFryq4qpAwkIARUAAIhCGAE=&rs=AOn4CLAsQEQayoWWnik8WVg35r2DUJO6gg" alt="YouTube Playlist">
  </a>
</p>

Check out the corresponding YouTube playlist for video tutorials on these techniques.

## Getting Started

Clone the repository and explore the notebooks to level up your deep learning skills.

```bash
git clone https://github.com/your_username/essential-tricks-in-deeplearning.git
cd essential-tricks-in-deeplearning
```

## Contributing

We welcome contributions from the community! If you have suggestions, bug reports, or want to add new tricks to the repository, follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/new-trick`.
3. Make your changes and commit: `git commit -m 'Add new trick: Feature Name'`.
4. Push to the branch: `git push origin feature/new-trick`.
5. Open a pull request.

## Resources

### Articles

- [Deep Learning Tips and Tricks cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks)
- [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-asynchronous-data-loading-and-augmentation)
- [Speed Up Model Training](https://lightning.ai/docs/pytorch/stable/advanced/speed.html)
- [EFFECTIVE TRAINING TECHNIQUES](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html)
- [Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/en/perf_train_gpu_one)
- [5 Must-Have Tricks When Training Neural Networks](https://deci.ai/blog/tricks-training-neural-networks/)
- [Deep Learning Tips and Tricks](https://towardsdatascience.com/deep-learning-tips-and-tricks-1ef708ec5f53)
- [Neural Network Tips and Tricks](https://thedatascientist.com/tips-tricks-neural-networks/)
- [A bunch of tips and tricks for training deep neural networks](https://towardsdatascience.com/a-bunch-of-tips-and-tricks-for-training-deep-neural-networks-3ca24c31ddc8)

### GitHub

- [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
- [Deep Learning Tricks](https://github.com/Conchylicultor/Deep-Learning-Tricks)
- [Deep Learning Training Tricks](https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/08-TransferLearning/TrainingTricks.md)
- [Tricks In Deep Learning](https://github.com/sherdencooper/tricks-in-deeplearning)
- [Tricks used in Deep Learning](https://github.com/bobchennan/tricks-used-in-deep-learning)
- [Efficient Deep Learning](https://github.com/Mountchicken/Efficient-Deep-Learning)
- [Deep Learning Tips and Tricks](https://github.com/ayyucedemirbas/Deep-Learning-Tips-and-Tricks)
- [Torch Memory-adaptive Algorithms (TOMA)](https://github.com/BlackHC/toma/tree/master)
- [Trading compute for memory in PyTorch models using Checkpointing](https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb)

### Papers

- [Bag of Tricks for Training Deeper Graph Neural Networks: A Comprehensive Benchmark Study](https://arxiv.org/abs/2108.10521)
- [Tricks From Deep Learning](https://arxiv.org/abs/1611.03777)

## To-Do
Extras - 
- [ ] Distributed Training
- [ ] TensorBoard Logging
- [ ] Adversarial training
- [ ] Autoscaling Batch Size
- [ ] Out of Fold Prediction
- [ ] Self Distillation
- [ ] OneCycleLR
- [ ] Snapshot ensembles
- [ ] Focal loss
- [ ] Drop path
