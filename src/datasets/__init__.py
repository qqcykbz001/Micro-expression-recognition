from src.datasets.casme2_dataset import CASME2Dataset
from src.datasets.samm_dataset import SAMMDataset

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
    elif dataset_name == 'samm':
        return SAMMDataset(config.root_dir, 
                          num_frames=config.num_frames, 
                          height=config.height, 
                          width=config.width, 
                          config=config, 
                          **kwargs)
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_name}")