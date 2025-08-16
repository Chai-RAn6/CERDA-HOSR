# CERDA-HOSR
   CERDA-HOSR is a novel computational method that leverages higher-order graph attention networks and graph convolutional networks to predict ceRNA-disease associations.

## Requirements
  * python==3.8
  * networkx==3.0
  * numpy==1.24.1
  * scikit-learn==1.3.0
  * tqdm==4.65.0
  * matplotlib==3.7.1
  * pandas==2.0.3
  * torch==2.0.0+cu118, torchaudio==2.0.1+cu118, torchvision==0.15.1+cu118

## Files
### Data
   data.xlsx: Dataset1 comprises 218 lncRNAs, 605 miRNAs, 1,051 mRNAs, and 314 human diseases, encompassing a total of 2,115 ceRNA network–disease regulatory associations. The dataset is publicly available and can be accessed through the LncACTdb database at http://bio-bigdata.hrbmu.edu.cn/LncACTdb.
  The data files needed to run the model, which contain CircR2Disease v1.0, CircR2Disease v2.0, HMDD v2.0, HMDD v3.0 and HMDD v4.0.
  * disease semantic similarity matrix 1.txt and disease semantic similarity matrix 2.txt: Two kinds of disease semantic similarity.
  * miRNA functional similarity matrix.txt: miRNA functional similarity.
  * known disease-miRNA association number.txt: Validated miRNA-disease associations.
  * disease number.txt: Disease ids and names.
  * miRNA number.txt: miRNA ids and names.

### Code
  * HOGAT.py: Example of Higher-Order attention process
  * training.py: Train the model
  * negative-cerna.py: High-order negative sampling process
  * main.py: running the complete training and evaluation pipeline
  * predict_horda.py: predict scores 
 
## Usage
  * Download code and data, take the CircR2Disease v1.0 dataset as an example:
  python predict_horda.py \
  --model_file best.pkl \
  --train_graph graph_train.txt \
  --full_graph CircR2Disease.txt \
  --rna_num 585 \
  --disease_num 88 \
  --topk 15
