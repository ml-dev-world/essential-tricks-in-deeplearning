# Essential Tricks In Deep Learning

<p align="center">
  <img src="https://assets.spe.org/dims4/default/7c6d2d6/2147483647/strip/true/crop/1051x552+0+0/resize/1200x630!/quality/90/?url=http%3A%2F%2Fspe-brightspot.s3.amazonaws.com%2F53%2F9d%2F228eca9b412bb1e3aa8b76d5f9db%2Fdaaiml.jpg" alt="Deep Learning">
</p>

Master key techniques to enhance your deep learning skills. This repository is a concise guide to essential tricks for optimizing model performance.

## Table of Contents

1. [Building Your First Neural Network](./01_Building_Your_First_Neural_Network.ipynb) - Introduction to neural networks for beginners.
2. [Pixels To Prediction Using ConvNet](./02_Pixels_To_Prediction_Using_ConvNet.ipynb) - Building and training your first Convolutional Neural Network.
3. [Dynamic Progress Bar Using tqdm](./03_Dynamic_Progress_Bar_Using_tqdm.ipynb) - Explore model initialization strategies and minor bug fixes.
4. [Decoding Dropout Technique](./04_Decoding_Dropout_Technique.ipynb) - Unravel the dropout technique and its impact on model training.
5. [Optimizing Convolutions With Batchnorm](./05_Optimizing_Convolutions_With_Batchnorm.ipynb) - Enhance model efficiency using batch normalization with convolutions.
6. [Exploring Model Initialization Strategies](./06_Exploring_Model_Initialization_Strategies.ipynb) - Dive into various strategies for initializing neural network weights.
7. [Memory Efficient DataLoader And Parallelization](./07_Memory_Efficient_DataLoader_And_Parallelization.ipynb) - Optimize data loading and processing for memory efficiency.
8. [Data Augmentation Using torchvision](./08_Data_Augmentation_Using_torchvision.ipynb) - Implement data augmentation techniques using the torchvision library.
9. [Improving Generalization With Label Smoothing](./09_Improving_Generalization_With_Label_Smoothing.ipynb) - Enhance model generalization using label smoothing.
10. [Ensuring Experiment Reproducibility](./10_Ensuring_Experiment_Reproducibility.ipynb) - Implement practices to ensure reproducibility in your experiments.
11. [Dynamic LR Scheduling](./11_Dynamic_LR_Scheduling.ipynb) - Implement dynamic learning rate scheduling techniques.
12. [Optimizing Learning Rate With LR Finder](./12_Optimizing_Learning_Rate_With_LR_Finder.ipynb) - Optimize learning rates using a learning rate finder.
13. [Warmup Strategies With Cosine Annealing](./13_Warmup_Strategies_With_Cosine_Annealing.ipynb) - Implement warmup strategies with cosine annealing for better convergence.
14. [Adaptive Gradient Clipping](./14_Adaptive_Gradient_Clipping.ipynb) - Explore gradient clipping and adaptive gradient clipping techniques.
15. [Controlling Overfitting With Weight Decay](./15_Controlling_Overfitting_With_Weight_Decay.ipynb) - Mitigate overfitting using weight decay and the AdamW optimizer.
16. [Smoothing Gradients With Gradient Penalty](./16_Smoothing_Gradients_With_Gradient_Penalty.ipynb) - Implement gradient penalty for smoother training gradients.
17. [Accumulating Gradient For Efficient Training](./17_Accumulating_Gradient_For_Efficient_Training.ipynb) - Optimize training efficiency by accumulating gradients.
18. [Automatic Mixed Precision Training](./18_Automatic_Mixed_Precision_Training.ipynb) -  Accelerate training by using a combination of lower-precision and higher-precision numerical representations.
19. [Effective Model Checkpointing](./19_Effective_Model_Checkpointing.ipynb) - Selecting the best-performing model.
20. [Early Stopping Strategies For Training](./20_Early_Stopping_Strategies_For_Training.ipynb) - Selection of a reasonably optimal model.
21. [Experiment Tracking With mlflow](./21_Experiment_Tracking_With_mlflow.ipynb) - mlflow for Experiment Tracking
22. [Logging Model Parameters](./22_Logging_Model_Parameters.ipynb) - Logging model parameters - flops, trainable params, size etc.
23. [Understanding CUDA Memory Usage](./23_Understanding_CUDA_Memory_Usage) - Manage GPU Usage.
24. [Hyperparameter Tuning With optuna](./24_Hyperparameter_Tuning_With_optuna.ipynb) - Automate the optimization process of hyperparameters.
25. [Hyperparameter Tuning With optuna - Training](./24_2_Hyperparameter_Tuning_With_optuna.ipynb) - Notebook demonstrating training with optimal hyperparams.

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
- [Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/en/perf_train_gpu_one)
- [5 Must-Have Tricks When Training Neural Networks](https://deci.ai/blog/tricks-training-neural-networks/)
- [Deep Learning Tips and Tricks](https://towardsdatascience.com/deep-learning-tips-and-tricks-1ef708ec5f53)
- [Neural Network Tips and Tricks](https://thedatascientist.com/tips-tricks-neural-networks/)

### GitHub

- [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
- [Deep Learning Tricks](https://github.com/Conchylicultor/Deep-Learning-Tricks)
- [Deep Learning Training Tricks](https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/08-TransferLearning/TrainingTricks.md)
- [Tricks In Deep Learning](https://github.com/sherdencooper/tricks-in-deeplearning)
- [Tricks used in Deep Learning](https://github.com/bobchennan/tricks-used-in-deep-learning)
- [Efficient Deep Learning](https://github.com/Mountchicken/Efficient-Deep-Learning)
- [Deep Learning Tips and Tricks](https://github.com/ayyucedemirbas/Deep-Learning-Tips-and-Tricks)

### Papers

- [Bag of Tricks for Training Deeper Graph Neural Networks: A Comprehensive Benchmark Study](https://arxiv.org/abs/2108.10521)
- [Tricks From Deep Learning](https://arxiv.org/abs/1611.03777)

## To-Do

- [ ] Deep Dive Into Error Analysis
- [ ] Understanding Confusion Matrix in Deep Learning
- [ ] Classwise Metrics for Model Evaluation
- [ ] Enhance Testing with Test Time Augmentation
- [ ] Improving Stability with Exponential Moving Average
- [ ] Stochastic Weight Averaging
- [ ] Gradient Checkpointing
- [ ] Adversarial Training
- [ ] Model Explainability
- [ ] Out of Fold Prediction
- [ ] TensorBoard Logging
- [ ] Autoscaling Batch Size
- [ ] Transfer Learning
- [ ] Differential Learning Rate
- [ ] Layerwise Learning Rate Decay
- [ ] Freeze / Unfreeze
- [ ] Progressive Scaling
- [ ] Self Distillation
- [ ] OneCycleLR
- [ ] Distributed Training
