
# Data directory structure:
Please organize the datasets under the `benchmark/RAW_DATA/` directory as described below. This structure follows the format used in the repository [missing_aware_prompts](https://github.com/YiLunLee/missing_aware_prompts).



## MM-IMDb
[MM-IMDb](https://github.com/johnarevalo/gmu-mmimdb) [(archive.org mirror)](https://archive.org/download/mmimdb)

    IMDB
    ├── images            
    │   ├── 00000005.jpeg 
    │   ├── 00000008.jpeg   
    │   └── ...        
    ├── labels          
    │   ├── 00000005.json 
    │   ├── 00000008.json   
    │   └── ...        
    └── split.json 


## Food101
[UPMC Food-101](https://visiir.isir.upmc.fr/explore) [(Kaggle)](https://www.kaggle.com/datasets/gianmarco96/upmcfood101?select=texts)

    FOOD101
    ├── images            
    │   ├── train                
    │   │   ├── apple_pie
    │   │   │   ├── apple_pie_0.jpg        
    │   │   │   └── ...         
    │   │   ├── baby_back_ribs  
    │   │   │   ├── baby_back_ribs_0.jpg        
    │   │   │   └── ...    
    │   │   └── ...
    │   ├── test                
    │   │   ├── apple_pie
    │   │   │   ├── apple_pie_0.jpg        
    │   │   │   └── ...         
    │   │   ├── baby_back_ribs  
    │   │   │   ├── baby_back_ribs_0.jpg        
    │   │   │   └── ...    
    │   │   └── ...
    ├── texts          
    │   ├── train_titles.csv            
    │   └── test_titles.csv         
    ├── class_idx.json         
    ├── text.json         
    └── split.json

```bash
# IMDB
mkdir -p benchmark/RAW_DATA/IMDB/generate_arrows
cd benchmark/RAW_DATA/IMDB/generate_arrows

# Food101
conda activate fmfl
mkdir -p benchmark/RAW_DATA/FOOD101
mkdir benchmark/RAW_DATA/FOOD101/missing_tables_8_classes/
mkdir benchmark/RAW_DATA/FOOD101/missing_tables_other_tests/
```