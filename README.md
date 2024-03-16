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
23. [Understanding CUDA Memory Usage](./23_Understanding_CUDA_Memory_Usage.ipynb) - Manage GPU Usage.
24. [Hyperparameter Tuning With optuna](./24_Hyperparameter_Tuning_With_optuna.ipynb) - Automate the optimization process of hyperparameters.
    - [Hyperparameter Tuning With optuna - Training](./24_2_Hyperparameter_Tuning_With_optuna.ipynb) - Notebook demonstrating training with optimal hyperparams.
25. [Deep Dive Into Error Analysis](./25_Deep_Dive_Into_Error_Analysis.ipynb) - How to calculate and evaluate your model.
26. [Understanding Confusion Matrix in Deep Learning](./26_Understanding_Confusion_Matrix_In_Deep_Learning.ipynb) - Implement confusion matrix.
27. [Classwise Metrics for Model Evaluation](27_Classwise_Metrics_For_Model_Evaluation.ipynb) - How to calculate class wise metrics.
28. [Stochastic Weight Averaging For Improved Convergence](28_Stochastic_Weight_Averaging_For_Improved_Convergence.ipynb) - Ensemble model via weight averaging.
29. [Memory Efficient Models With Checkpointing](29_Memory_Efficient_Models_with_Checkpointing.ipynb) - Efficient memory usage in training.
30. [Enhancing Testing with Test Time Augmentation](30_Enhancing_Testing_With_Test_Time_Augmentation.ipynb) - Enhancing model predictions with augmentation.
31. [Model Interpretibilit With captum](31_Model_Interpretibility_With_captum.ipynb) - Exploring Model Insights with Interpretability.
32. [Going Deeper with Transfer Learning](32_Going_Deeper_With_Transfer_Learning.ipynb) - Leverage knowledge gained from solving one problem to improve performance on another problem.
33. [Freeze Unfreeze Backbone](33_Freezing_Backbone.ipynb) - Selectively enabling or disabling the training of specific layers
34. [Differential Learning Rate](34_Differential_Learning_Rate.ipynb) - Different learning rates are applied to different parameters.
35. [Layerwise LR Decay](./35_Layerwise_Learning_Rate_Decay.ipynb) -  Training deep neural networks to adjust the learning rate for each layer individually.

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

- [x] Deep Dive Into Error Analysis
- [x] Understanding Confusion Matrix in Deep Learning
- [x] Classwise Metrics for Model Evaluation
- [x] Stochastic Weight Averaging
- [x] Gradient Checkpointing

// 13.3.2024
- [x] Enhance Testing with Test Time Augmentation
- [x] Model Explainability

// 14.3.2024
- [x] Transfer Learning
- [x] Differential Learning Rate
- [x] Layerwise Learning Rate Decay
- [x] Freeze / Unfreeze

// 15.3.2024
- [ ] Improving Stability with Exponential Moving Average
- [ ] TensorBoard Logging

// 16.3.2024
- [ ] Adversarial Training
- [ ] Autoscaling Batch Size
- [ ] Progressive Image Resizing

// 17.3.2024
- [ ] Out of Fold Prediction
- [ ] Self Distillation
- [ ] OneCycleLR
- [ ] Distributed Training
