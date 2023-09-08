# PyTorch 深度学习的实验
PyTorch deep learning project made easy.

## 环境
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

## 功能简介
* 清晰的目录结构适合各种深度学习的项目
* json配置文件支持方面的调整参数
* 自定义命令行参数
* 支持训练过程中的数据的保存和使用
* 形象定义的类方便快速的开发新的功能
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary.

## 目录结构
  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  ├── config
  │   ├── config.json - holds configuration for training
  │   └── parse_config.py - class to handle config file and cli options
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

## 如何训练使用

运行命令 `python train.py -c config/config.json` 进行训练.

### 修改配置文件
```javascript
{
  "name": "Mnist_LeNet",        // 训练系列的名称
  "n_gpu": 1,                   // 训练的时候使用的gpu的个数
  
  "arch": {
    "type": "MnistModel",       // 训练的模型的框架的名字
    "args": {}
  },
  "data_loader": {
    "type": "MnistDataLoader",         // 选择 data loader
    "args":{
      "data_dir": "data/",             // dataset 的路径
      "batch_size": 64,                // batch size 的大小
      "shuffle": true,                 // 时候打乱数据
      "validation_split": 0.1          // 验证数据的大小. float(portion) or int(number of samples)
      "num_workers": 2,                // 加载数据时候的线程数量
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // 学习率
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": "nll_loss",                  // 损失
  "metrics": [
    "accuracy", "top_k_acc"            // 评估指标
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 10	                 // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,               // enable tensorboard visualization
  }
}
```
### 训练重新开始
你可以选择重新开始训练从先前训练好的模型数据:
  ```
  python train.py --resume path/to/checkpoint
  ```

### 使用多个GPU
  ```
  python train.py --device 2,3 -c config.json
  #等价于:
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

## 新增实验项

### 新增命令行参数
通过配置文件修改超参数是简单和方便的，但是对于经常修改的参数最好可以支持命令行的方式输入,可以通过下面的方式自定义需要命令行输入的参数,其中的target指的是配置文件的dict参数的一个序列：optimizer->args->lr: python train.py --lr 0.01

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```

### 加载数据
* **如何自定义开发自己的加载数据的功能**

1. **继承 ```BaseDataLoader```**

    `BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.

    `BaseDataLoader` 包含方法:
    * 生成下一个 bach
    * Data shuffling
    * 生成 validation data loader by calling
    `BaseDataLoader.split_validation()`

* **DataLoader 使用**

  `BaseDataLoader` 是一个迭代器,迭代batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **例子**

  Please refer to `data_loader/data_loaders.py` for an MNIST data loading example.

### 训练
- **实现自定义的训练**

  1.**派生自 ```BaseTrainer```**
    `BaseTrainer` 需要实现:
    * Training process logging
    * Checkpoint saving
    * Checkpoint resuming
    * Reconfigurable performance monitoring for saving current best model, and early stop training.
      * If config `monitor` is set to `max val_accuracy`, which means then the trainer will save a checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
      * If config `early_stop` is set, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.

    2.**抽象的接口实现**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`

- **Example**

  Please refer to `trainer/trainer.py` for MNIST training.

- **迭代训练**

  `Trainer.__init__` takes an optional argument, `len_epoch` which controls number of batches(steps) in each epoch.

### 自定义模型
* **实现定定义模型**

1. **基类 `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **抽象方法实现**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/model.py` for a LeNet example.

### 自定义损失函数
自定义的损失函数在 'model/loss.py'中实现. 可以通过在config文件中修改loss: name 配置它.

### 自定义Metric
关于指标的函数实现在文件'model/metric.py'中.

您可以通过在配置文件中提供列表来监控多个指标, e.g.:
  ```json
  "metrics": ["accuracy", "top_k_acc"],
  ```

### 自定义log信息

如果有需要打印的中间过程信息, 在 训练类中的方法`_train_epoch()`,更新log,如下:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```

### 验证
您可以通过运行“test.py”来测试经过训练的模型，通过“--resume”传递训练过的checkpoint路径。

### 评估数据
要从数据加载器中拆分验证数据，请调用“BaseDataLoader.split_validation()”，然后它将返回一个数据加载器。 `validation_split` 可以是验证集与总数据的比率 (0.0 <= float < 1.0)，或样本数量 (0 <= int < `n_total_samples`)。

**Note**: `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints
您可以在配置文件中指定训练的名称：
  ```json
  "name": "MNIST_LeNet",
  ```

The checkpoints 将会保存到 `save_dir/name/timestamp/checkpoint_epoch_n`, 会加上相应的时间戳.同时会把配置文件也保存到文件夹中.

**Note**: checkpoints 包含:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```

### Tensorboard 可视化

此模板通过使用“torch.utils.tensorboard”或 [TensorboardX](https://github.com/lanpa/tensorboardX) 支持 Tensorboard 可视化。

1. **安装**
    ```shell
    pip install tensorboard>=1.14.0
    ```
2. **使用** 
    配置文件中进行设置选项 `tensorboard`.
    ```shell
     "tensorboard" : true
    ```
3. **开启服务** 

    在项目根目录输入命令“tensorboard --logdir saving/log/”，然后服务器将在“http://localhost:6006”打开

默认情况下，将记录输入图像和模型参数直方图, 损失和指标。如果您需要更多可视化，请在`trainer._train_epoch`方法中使用`add_scalar('tag', data)`、`add_image('tag', image)`等。

该模板中的“add_something()”方法基本上是“tensorboardX.SummaryWriter”和“torch.utils.tensorboard.SummaryWriter”模块的包装器。

**注意**：您不必指定当前步骤，因为在 `logger/visualization.py` 中定义的 `WriterTensorboard` 类将跟踪当前步骤。
