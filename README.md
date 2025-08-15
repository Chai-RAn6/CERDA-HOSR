# CERDA-HOSR
CERDA-HOSR is a novel computational method that leverages higher-order graph attention networks (GATs) and graph convolutional networks (GCNs) to predict ceRNA-disease associations.

## Requirements
  * python==3.8
  * networkx==3.0
  * numpy==1.24.1
  * scikit-learn==1.3.0
  * tqdm==4.65.0
  * matplotlib==3.7.1
  * pandas==2.0.3
  * torch==2.0.0+cu118 torchaudio==2.0.1+cu118 torchvision==0.15.1+cu118


## Files
### data
  The data files needed to run the model, which contain HMDDv2.0 and HMDDv3.2.
  * disease semantic similarity matrix 1.txt and disease semantic similarity matrix 2.txt: Two kinds of disease semantic similarity
  * miRNA functional similarity matrix.txt: MiRNA functional similarity
  * known disease-miRNA association number.txt:Validated mirNA-disease associations
  * disease number.txt: Disease id and name
  * miRNA number.txt: MiRNA id and name

### code
  * eval.py: The startup code of the program
  * training.py: Train the model
  * negative-cerna.py: High-order negative sampling process
 
## Usage
  * Download code and data, take the CircR2Disease v1.0 dataset as an example:
  python predict_horda.py \
  --model_file best.pkl \
  --train_graph graph_train.txt \
  --full_graph CircR2Disease.txt \
  --rna_num 585 \
  --disease_num 88 \
  --topk 15
