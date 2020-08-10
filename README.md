# SF-Net: Single-Frame Supervision for Temporal Action Localization

> [**SF-Net**](https://arxiv.org/abs/2003.06845),            
> Fan Ma, Linchao Zhu, Yi Yang, Shengxin Zha, Gourab Kundu, Matt Feiszli, Zheng Shou        
> *arXiv technical report ([arXiv 2003.06845](https://arxiv.org/abs/2003.06845))*  


    @article{ma2020sfnet,
    title={SF-Net: Single-Frame Supervision for Temporal Action Localization},
    author={Fan Ma and Linchao Zhu and Yi Yang and Shengxin Zha and Gourab Kundu and Matt Feiszli and Zheng Shou},
    year={2020},
    eprint={2003.06845},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }

Contact: [fan.ma@student.uts.edu.au](mailto:fan.ma@student.uts.edu.au). Any questions or discussion are welcome! 

## Abstract
In this paper, we study an intermediate form of supervision, i.e., single-frame supervision, for temporal action localization (TAL). To obtain the single-frame supervision, the annotators are asked to identify only a single frame within the temporal window of an action. This can significantly reduce the labor cost of obtaining full supervision which requires annotating the action boundary. Compared to the weak supervision that only annotates the video-level label, the single-frame supervision introduces extra temporal action signals while maintaining low annotation overhead. To make full use of such single-frame supervision, we propose a unified system called SF-Net. First, we propose to predict an actionness score for each video frame. Along with a typical category score, the actionness score can provide comprehensive information about the occurrence of a potential action and aid the temporal boundary refinement during inference. Second, we mine pseudo action and background frames based on the single-frame annotations. We identify pseudo action frames by adaptively expanding each annotated single frame to its nearby, contextual frames and we mine pseudo background frames from all the unannotated frames across multiple videos. Together with the ground-truth labeled frames, these pseudo-labeled frames are further used for training the classifier. In extensive experiments on THUMOS14, GTEA, and BEOID, SF-Net significantly improves upon state-of-the-art weakly-supervised methods in terms of both segment localization and single-frame localization. Notably, SF-Net achieves comparable results to its fully-supervised counterpart which requires much more resource intensive annotations. 

## Features 
All features can be downloaded from the [drive](https://drive.google.com/drive/folders/1DfLDau7hqb-5huhB3W-3XljeuFu2YcF9?usp=sharing). Put the features into the data directory. 


## Run on THUMOS14 
~~~
python main.py
~~~

### Results on THUMOS14

|  mAP@0.3  | mAP@0.3 | mAP@0.5 |
|-----------|---------|---------|
| 53.04     |   29.82 | 10.87    |
