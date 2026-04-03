from src.datasets.casme2_dataset import CASME2Dataset

def get_dataset(config, **kwargs):
    """数据集工厂函数"""
    dataset_name = config.dataset_name.lower()
    
    if dataset_name == 'casme2':
        return CASME2Dataset(config.root_dir, 
                            num_frames=config.num_frames, 
                            height=config.height, 
                            width=config.width, 
                            config=config, 
                            **kwargs)
    # 未来可以在这里添加其他数据集，例如:
    # elif dataset_name == 'samm':
    #     return SAMMDataset(...)
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_name}")
