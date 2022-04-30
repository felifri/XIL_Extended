# Results ISIC Skin Cancer 2019
## Experimentation setup
About the ISIC Skin Cancer 2019 dataset:  
* has two classes: Benign (Not-Cancer) and Malignant (Cancer) 
* Class distribution: Not-Cancer= 19735; Cancer= 2557 (ratio 88%:12%)
* Not-Cancer class contains confounding factor (patch) in 9210 images: Ratio 47%
* Cancer images DO NOT contain any patches

In order to check wether the patches are used as informative feature we constructed two different test sets.  
1. Test set patches: Contains only not-cancer images **with** patches plus the cancerous images.
2. test set no patches: Contains only not-cancer images **without** patches plus the cancerous images.
The cancerous images in both test sets are identical.   
In order to evaluate the performance of XIL methods we generated feedback annotation mask which indicated the region of the patches.  

If the model used the patches as informative featire to distinguish between the two classes, we expect a better performance for the vanilla model (without XIL) on the test set with patches - as the model trained on images with patches also. The performance should drop on the test set with no patches. More specifically, we expect a lower recall (sensitivity) for the not-cancer class (i.e. more False Positives - classifying not-cancer images as cancerous) and consequently also a lower precision for the cancer class.

### Dataset factsheet
Labels: 0 -> not_cancer; 1 -> cancer  

#### Sizes of datasets (ratio):  
* **train**: 17829 (not_cancer=15768 [**88%**]; cancer=2061 [**12%**])
    - Ratio of patches/no_patches: no_patches=10443 [**59%**]; patches=7386 [**41%**]
    - Note: patches only occure in not_cancer images (in 7386 of 15768) -> 47% 
* **test patches**: 2316 (not_cancer=1824 [**79%**]; cancer=492 [**21%**])
* **test no patches**: 2634 (not_cancer=2142 [**81%**]; incancer=492 [**19%**]) 

As common in medical classification tasks, the dataset has a large class imbalance in favor of benign examples. Only 12% of the training examples have a malignant diagnosis. We account for the class imbalance by applying a weighted loss during the optimization process. A classifier which classifies all samples as benign would reach an accurcay of 88% on the train set (79% on test patches, 81% on test set no patches).

# Baselines
## Model configuration
Model: VGG16 pretrained (pytorch) with last classifier replaced to predict binary class.  
Hyperparams:
* epochs: 50
* batch_size: 16 (could not incease beacause of GPU Memory)
* optimizer: SGD (momentum=0.9)
* lr: 0.001
* scheduler: ReduceLROnPlateau(train_loss, patience=8)
* seed: 100, 10 (avg)
* train_shuffle: True
    
## Vanilla model 
**Train acc: 100.0%**, **train loss: 0.000001**

### Test set patches
Acc: 92.15%, Loss: 0.050214

**Confusion matrix:** 
in brackets: seed=10 runs
| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1824       | **0**  |
| cancer     | 181  (182) | 311 (310)|  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.91      |**1.00**| 0.95     | 1824    |
| cancer      | **1.00**  | 0.63   | 0.77     | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.92     | 2316    |
| macro avg   | 0.95      | 0.82   | **0.86** | 2316    |
| weighted avg| 0.93      | 0.95   | 0.91     | 2316    |


### Test set no patches
Acc: 87.65%, Avg loss: 0.0650225

**Confusion matrix:** 
in brackets: seed=10 runs
| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 2005 (1991)| **137** (151)|
| cancer     | 181 (182)       | 311  (310)  |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.92      |**0.94** (0.93)| 0.93  (0.92)   | 2142    |
| cancer      | **0.69** (0.67) | 0.63   | 0.66  (0.65)   | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.88     | 2634    |
| macro avg   | 0.81 (0.79)     | 0.78   | **0.79** | 2634    |
| weighted avg| 0.93  (0.87)    | 0.95  (0.87) | 0.88  (0.87)   | 2634    |

### Interpretation

In the table aboves we can see that the accurcay drops from 92% on the test set with patches to 88% on the tes without patches. Upon closer examination of the confusion matrix we see that on the test set without patches our model yields more False Positives (0 -> 137) - so more actual benign images get missclassified as cancerous images. We follow that it is more difficult for the model to identify the not-cancer class, because the malignant images are identical in both sets. This is captured in the performance metric of recall for the not-cancer class, dropping by 6% from 100% to 94%. Consequently also the precision of the cancer class dropping by 31% from 100% to 69%. A good comparison is the macro average f1-score. The test set with patches results in an macro-avg f1 of 86%, in contrast the test set without patches yields a macro-avg f1 of 79%, thats a difference of 7%.

To check if the vanilla model truly relies on the confouding factors - the patches - we also have to inspect some visualizations of explainer methods for specific predictions.


## RRR 
params; reg_rate=10, weighted loss  

**Train acc: 98.77%**, **train loss: 0.005165**
98.2% 0.006622

**AVG train acc: 98.485%**, **train loss: 0.0058935**

### Test set patches
Acc: 91.7%, Loss: 0.020020 
AVG Acc: 93.1% (+-1.2), Loss: 0.01675

**Confusion matrix:** 
in brackets run with seed=10
| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1821 (1809)| **3** (15) |
| cancer     | 189  (113) | 303   (379)|  


**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.91 (0.94)     |  1.00 (0.99) | 0.95  (0.97)   | 1824    |
| cancer      | 0.99  (0.96)    |  0.62 (0.77) | 0.76  (0.86)   | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.92 (0.94)    | 2316    |
| macro avg   | 0.95      | 0.81 (0.88)  | 0.85 (0.91)    | 2316    |
| weighted avg| 0.92  (0.95)    | 0.92  (0.94) | 0.91  (0.94)   | 2316    |


### Test set no patches
Acc: 87.6%, Avg loss: 0.026498 
Acc: 81.6%, Avg loss: 0.037889
AVG: Acc: 84.6% (+-3.0), Loss: 0.03219

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 2004 (1770)| **138** (372)|
| cancer     | 189  (113)      | 303 (379)|  


**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.91 (0.94)     |**0.94** (0.83)| 0.92 (0.88)    | 2142    |
| cancer      | **0.69** (0.5) | 0.62 (0.77)  | 0.65 (0.61)    | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.88     | 2634    |
| macro avg   | 0.81 (0.72)     | 0.78 (0.8)  | **0.79** (0.74)| 2634    |
| weighted avg| 0.87  (0.86)    | 0.88 (0.82)  | 0.87  (0.83)   | 2634    |

### Interpretation statistics
RRR with reg rate 10 does not improve performnace on the test set with no patches. In comparison to the vanilla model performance is nearly identical on both test sets.
**Hypothesis**:  
* maybe the influence of the right reason loss is too small because of a small reg rate. We compared the avg ra and rr losses -> ra=0.0137 vs. rr=0.00204. --> reg rate of 100 would squeeze losses into same order of magnitude. --> tried it but no improvement  
* train longer than 50 epochs. Currently only 98.77 % train acc vs. 100% on vanilla model  
* the patches are too important for the ability of a VGG16 to learn to classify ISIC. In other words, the dataset is to difficult. Penalizing the patches therefore does not help the model to figure out the true causal features (check explanations heatmaps to see wether the XIL model does not focus on the patches anymore).  
* penalizing gradients to overcome confounder does not work. Interestingly, the performance (recall) on the test set with patches does not drop in comparsion with the vanilla model, indicating that RRR does not hurt the model either. --> widerlegt mittels Visualisierungen


## RRRGradCAM
params; reg_rate=1, weighted loss, rr_clipping=10.0  

**Train acc: 100.0%**, **train loss: 0.000022**

### Test set patches
Acc: 92.0%, Loss: 0.041700
seed=10 Acc:92.0%, Loss: 0.048804

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1824       |   0    |
| cancer     | 186        | 306    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.91      |  1.00  | 0.95     | 1824    |
| cancer      | 1.00      |  0.62  | 0.77     | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.92     | 2316    |
| macro avg   | 0.95      | 0.81   | 0.86     | 2316    |
| weighted avg| 0.93      | 0.92   | 0.91     | 2316    |


### Test set no patches
Acc: 87.6%, Avg loss: 0.026498
seed=10 Acc: 87.2%, loss: 0.062690 

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1938 (1992)| 204 (150)|
| cancer     | 186        | 306    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.91      |**0.90** (0.93)| 0.91 (0.92)    | 2142    |
| cancer      | **0.60** (0.67) | 0.62 (0.62)  | 0.61 (0.65)    | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.85 (0.87)     | 2634    |
| macro avg   | 0.76  (0.79)    | 0.76 (0.78)  | **0.76** (0.78)| 2634    |
| weighted avg| 0.85  (0.87)    | 0.85  (0.87) | 0.85  (0.87)   | 2634    |


## CDEP
params; reg_rate=10, weighted loss, rr_clipping=10.0  

Note: had to decrease batch size to 4 because of huge GPU requiremnet of CDEP

**Train acc: 100.0%**, **train loss: 0.000002**

### Test set patches
Acc: 91.9%%, Loss: 0.205060

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1824       |   0    |
| cancer     | 188        | 304    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.91      |  1.00  | 0.95     | 1824    |
| cancer      | 1.00      |  0.62  | 0.76     | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.92     | 2316    |
| macro avg   | 0.95      | 0.81   | 0.86     | 2316    |
| weighted avg| 0.93      | 0.92   | 0.91     | 2316    |


### Test set no patches
Acc: 88.1%%, Avg loss: 0.249683 

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 2016       | 126    |
| cancer     | 188        | 304    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.91      |**0.94**| 0.93     | 2142    |
| cancer      | **0.71**  | 0.62   | 0.66     | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.88     | 2634    |
| macro avg   | 0.81      | 0.78   | **0.79** | 2634    |
| weighted avg| 0.88      | 0.88   | 0.88     | 2634    |

### Inspection of the visualized explanations

## CE
params; ce_strategy=random, n_counterexamples_per_instance=1, n_instances=7386 (all), weighted loss=0.0860->class=0, 0.9140->class=1   
TRAIN: 25215, TEST: 2316, TEST_NO_PATCHES: 2634
TRAIN class dist: Counter({0: 23154, 1: 2061})
TRAIN patch dist: Counter({0: 10443, 1: 7386, -1: 7386}) 

**Train acc: 100.0%**, **train loss: 0.000001**
seed= train acc: 100.0 
### Test set patches
Acc: 92.1%%, Loss: 0.047570
seed: 91.8%, 0.045932
**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1823       |   1    |
| cancer     | 181        | 311    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.91      |  1.00  | 0.95     | 1824    |
| cancer      | 1.00      |  0.63  | 0.77     | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.92     | 2316    |
| macro avg   | 0.95 ( 0.95)     | 0.82 (0.81)  | 0.86  (0.86)   | 2316    |
| weighted avg| 0.93      | 0.92   | 0.91     | 2316    |


### Test set no patches
Acc: 87.7%%, Avg loss: 0.061619 
seed 10: 86.8%, 0.062003

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1998       | 144    |
| cancer     | 181        | 311    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.92      |**0.93**| 0.93     | 2142    |
| cancer      | **0.68**  | 0.63   | 0.66     | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.88     | 2634    |
| macro avg   | 0.80 (0.78)      | 0.78 (0.77)  | **0.79**(0.78) | 2634    |
| weighted avg| 0.87      | 0.88   | 0.88     | 2634    |

### Inspection of the visualized explanations
TODO

## HINT
params: reg_rate=1, epochs= 50, batch_size=16, lr=0.001, optim=SGD(momentum=0.9), seeds=100, scheduler=True, shuffle=True
**Train acc: 98.97%** (epoch 48 -> 100%), **train loss: 0.001829**
seed=10 -> train=100%, 0.000001

### Test set patches
Acc: 90.3%, Loss: 0.032504
seed=10 -> 92.1, 0.045758 

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1824       | **0**  |
| cancer     | 224        | 268    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.89      | 1.00   | 0.94     | 1824    |
| cancer      |   1.00    |**0.54**| 0.71     | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.90     | 2316    |
| macro avg   | 0.95 (0.95)|**0.77** (0.81)| **0.82** (0.86)| 2316    |
| weighted avg| 0.93      | 0.95   | 0.91     | 2316    |


### Test set no patches
Acc: 86.0%, Avg loss: 0.037463 
87.7, 0.058927

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1996       | **146**|
| cancer     | 224        | 268    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.90      |  0.93  | 0.92     | 2142    |
| cancer      | **0.65**  | 0.54   | **0.59** | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.86     | 2634    |
| macro avg   | 0.77 (0.80)     | 0.74 (0.78)  | 0.75 (0.79)    | 2634    |
| weighted avg| 0.85      | 0.86   | 0.85     | 2634    |

### Interpretation

## Analysis of heatmaps
### Vanilla

Gradcam:
* in general GradCAM visualization confirm that the patches are used as informative feature in almost every image which contains a patch. 
* often region around the border of the patch (with pixels inside and outside the patch) highly is activated, indicating that the patches have a huge influnece on the prediction. 
* For images containig a patch the activation almost never highlighted the area of the lesion - in contrast images that had no patch showed signficant more activation in the lesion region. 
* dark corners and darker image borders also quite frequent showed increased activation, constiting another confounder.  

IG:
* IG heatmaps confimred that the vanilla model used the patches as relevant feature in almost every case. Especially the borders of the patches were visible. However mots of the time also the the lesion was activated. 

### RRR
Gradcam:
* GradCAM visualization confirm that the RRR model does not focus on the patches anymore in most of the cases. But we also encountered some images where the RRR model highlighted the patches and in such cases huge parts of the patch were very strongly activated. This could explain the relatively big GradCAM WR score of RRR. (It either gets it or it completely focus on the confounder) 
* image with patches most often show white cutouts around the patches, indicating that the influence of the confounder was less comparaed to the vanilla model. Region of the lesion showed more activation. However also the darker borders/corners had a higher chance to be activated (Maybe to Reward vs. Penalize section: In a dataset that has more than one confoudner -> telling the model what's wrong does not necessarily guide the model to give right explanations. It just reduces the influence of the penalized one. Can also increase the chance that other unknown confoudner get exploited by the model, but this on the other hand can promote the confounder detection in a dataset if properly analyized.)

IG:
* showed significantly less activation on the patches and highlighted the region of the lesion.  


### RRR-G
GradCAM:
* heatmaps show signficanly less general activation. Only small areas or scattered pixel areas were acivated, especially on the images containing patches. More difficult to pinpoint the activation back to the orginal image, because of the upscaling we encountered inaccuracies.
* GradCAM visualization confirm that the RRRGradCAM model does not focus on the center of the patches anymore in most of the cases. In many cases the focus is still near the patches border, indicating that areas close the patch border (not covered by the masks) are still relevant. We hypothesize that the the masks needs to be bigger and maybe in the downscaling process of the mask (to multiply it with the output of the last conv layer) regions close to teh patch border influence the downscaled mask. Dark corners and borders on the images also constitued confounders as well.
* most of the time the area of the lesion is not activated.

IG:
* patches are almost never activated, but instead the dark top left corner was activated very frequently - as already mentioned these dark corners also constitued a confounder which we didi not penelize with masks.  


### CDEP
Gradcam:
* in general strongest activation across the whole image or large image regions compared to other methods. We detected a pattern of claer white cutouts with sharp borders in the patches region and also in many cases the true lesion area also showed these white cutouts, which is contraprductive to the real objective. Furthermore thr adjoining border pixels of the patch ost often were strongly activating. We followed, that CDEP rather learns to indentify large contiguous areas in the image (patches as well as bigger lesions) and penalizes them. Considering the influnece of the adjoining border pixels we followed that it did not help to reduce the influence.  

IG:
* not real difference to Vanilla heatmaps, patches are still always activated. In general the activation was very scattered across the image also in regions which were netiher the patch nor the lesion. (confirmed by WR metric)    

### HINT
GradCAM:
* in general very sparse activation. Almost no activation of the patches anymore, instead almost anytime focused on the real skin leasions (only on images which had confounder). The other confounders (dark corners and borders) which influenced RRR, RRR-G, are almost never activated.
* promising, but IG and LIME WR metric do not indicate a great decarese? [CHECK other heatmaps and combine faithfullness?]

IG:
* In contrast to the GradCAM heatmaps showed also activation on the patches even though not on the whole patch but instead on smaller cutout on the edges. Still conflicting with GradCAM explanations. 


### CE
GradCAM:
* more frequent the activation was on the border of the image. General very scattered. Probably less on the patches, sometimes on the border of the patches, but also not on the skin lesions. Not very informative.  

IG:
* relatively similar to the HINT heatmaps. 

## Quantification Wrong Reason (on test set with patches)
In order to quantify the impact of the specified XIL method on the predictions of the model, we calculated the wrong reason case as follows:  
Before quantifying wrong reasons we binarized the activation heatmap replacing all pixels bigger (smaller) than a precalculated median/mean with 1.0 (0.0).  
We calculated the normalized positive amount of activation (with a specific explainer method) an image has in the patches region and divided it through the maximal possible activation. This gets us a percentage for each image. Summing up all images and dividing by the total number of images (we only take images with patches in account), we get an overall percentage for a specific XIL method on the ISIC test set with patches. Comparing this score to the the score the vanilla model has gives us an estimation of how well the XIL methods helps to overcome the focus on the confounder. Note that we do not take the close sorrounding of the patches into account, which we saw in the visualizations does have an impact on the prediction even when the main area of the patches does not.  We also expect a specific XIL method to perform best on their utilized explanation method, i.e. lower percentage on their used explanation method.  

-> Note that we do not take the close sorrounding of the patches into account, which we saw in the visualizations does have an impact on the prediction even when the main area of the patches does not. We also expect a specific XIL method to perform best on their utilized explanation method, i.e. lower percentage on their used explanation method.


### Results Binarize MEDIAN
How did we calculated the Median: For every attribution heatmap of instances with patches we calculat the median pixel value (pixel values are normalized bewtween 0-1 beforehand). This gives us a list of median pixel values. We then return the median of the list of medians. 
--> brackets (std): Percentage is avg over all instances with a patch (excluding all zero attribution heatmpas).

| model \ expl  | GradCAM            | IG (Ross)        | Lime           |
|-------------- |--------------------|------------------|----------------|
| Vanilla       | 20.03% (±16.12)    | 50.13% (±0.46)   | 65.70% (±16.95)|
| RRR (10)      | 30.54% (±35.01)    | 49.53% (±0.99)   | 51.64$ (±20.72)|
| RRRGradCAM (1)| 00.29% (±2.7)\*      | 49.94% (±0.55)   | 64.26% (±15.45)|
| CDEP (10)     | 15.73% (±18.99)    | 49.99% (±0.47)   | 67.59% (±14.12)|
| CE            | 13.23% (±15.55)    | 49.88% (±0.52)   | 66.73% (±17.40)|
| HINT          | 0.885% (±4.75)\*\*     | 50.05% (±0.48)   | 64.40% (±14.11)|

### Results Binarize MEAN

| model \ expl  | GradCAM        | IG (Ross)        | Lime           |
|-------------- |----------------|------------------|----------------|
| Vanilla       |18.3% (±14.74)  | 32.49% (±7.14)   | 53.18% (±18.35)|
| RRR (10)      |24.47% (±32.28) | 12.88% (±13.02)  | 34.91% (±22.38)|
| RRRGradCAM (1)|0.056% (±1.00)\*| 13.69% (±7.46)   | 52.71% (±19.72)|
| CDEP (10)     |14.61% (±18.30) | 29.90% (±6.52)   | 57.00% (±16.24)|
| CE            |11.89% (±13.6)  | 29.87% (±8.72)   | 55.49% (±20.74)|
| HINT          |0.832% (±4.38)\*\*| 31.80% (±7.72)   | 52.96% (±16.61)|

\* 81 of 1824 attribution heatmpas were completely zero
\*\* 665 of 1824 attribution heatmpas were comletely zero

### Vanilla
* IG: Number img: 1824, abs attr: 27.62, avg_attr_per_img: **0.01515** -> **1.5%**
* Gradcam: Number img: 1824, abs attr: 153.11, avg_attr_per_img: **0.08394** -> **8.3%** 
* Lime: Number img: 86, abs attr: 30.39, avg_attr_per_img: **0.35334** -> **35.33%**

#### Gradcam:
Median:
Model ISIC19-vanilla-seed=100 loaded! Was trained on CrossEntropyLoss() for 50 epochs!
MEDIAN = 0.0 [MEDIAN WOLF Version= 0.0]
Number of actScores= 1824
Activation AVG per instance = 20.029688291667792 %
STD = 16.122938470845156 %
Activation ABS sum = 365.34151444002055
Number of complete zero attr= 0

Mean: 
MEAN= 0.040616141734700933:
Activation AVG per instance = 18.29992247536817
STD = 14.73648123334646
Activation ABS sum = 333.79058595071547

#### IG
Median:
MEDIAN= 1.8482975974620786e-05 [wolf version 1.641678136365954e-05]
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 50.13060499832296 %
STD = 0.46266821561061827 %
Activation ABS sum = 914.3822351694107

Mean:
MEAN= 0.009769760561124585
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 32.49235184773345
STD = 7.1410267856226035
Activation ABS sum = 592.6604977026582

#### Lime
Median:
median= 0.043823858723044395
Number of actScores= 1816
Number of complete zero attr= 8
Activation AVG per instance = 65.69521578032901
STD = 16.95843701348555
Activation ABS sum = 1193.0251185707748

Mean: 
mean = 0.15315473601139984
Number of actScores= 1816
Number of complete zero attr= 8
Activation AVG per instance = 53.17741931100183
STD = 18.34991028477902
Activation ABS sum = 965.7019346877933



### RRR (reg=10)
* IG: Number img: 1824, abs attr: 5.86, avg_attr_per_img: **0.003213** -> **0.3%**
* Gradcam: Number img: 1824, abs attr: 210.34, avg_attr_per_img: **0.11535** -> **11.5%**
* Lime: Number img: 86, abs attr: 15.68, avg_attr_per_img: 0.18239 --> **18.23%**

#### Gradcam
Median:
Model ISIC19-RRR-reg=10-seed=100 loaded! Was trained on RRRLoss for 0 epochs!
MEDIAN= 0.0 [wolf version 0.0]
Number of actScores= 1824;
Activation AVG per instance = 30.536404160598625 %
STD = 35.01041233347535 %
Activation ABS sum = 556.9840118893189
Number of complete zero attr= 0

Mean:
MEAN= 0.10493943087344074
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 24.467576303226323
STD = 32.28070886422757
Activation ABS sum = 446.28859177084814

#### IG
Median:
MEDIAN= 9.457632586418185e-06 [wolf version 9.061025593837257e-06]
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 49.53163941309117 %
STD = 0.9864126853257832 %
Activation ABS sum = 903.457102894783

Mean:
MEAN= 0.006411466257161105
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 12.881102067635247
STD = 13.021295783492986
Activation ABS sum = 234.9513017136669
Number of complete zero attr= 0

#### Lime
Median:
median= 0.07386811822652817
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 51.64173916998327
STD = 20.71825245564049
Activation ABS sum = 941.9453224604949

Mean:
mean= 0.17508005321213746
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 34.90541306724928
STD = 22.383719024872406
Activation ABS sum = 636.6747343466268


### RRRGradCAM (reg=1)
* IG: Number img: 1824, abs attr: 7.48, avg_attr_per_img: **0.00409** -> **0.41%**
* Gradcam: Number img: 1824, abs attr: 0.125, avg_attr_per_img: **0.000068** -> **0.0068%**
* Lime: Number img: 85, abs attr: 32.74, avg_attr_per_img: 0.38518 --> **38.51%**

#### Gradcam 
Median:
Model ISIC19-RRRGradCAM-reg=1-seed=100-clip=10-c loaded! Was trained on RRRGradCamLoss
Median= 0.0
Number of actScores= 1743
Activation AVG per instance = 0.28987338620823794 %
STD = 2.7121359521801343 %
Activation ABS sum = 5.0524931216095865
Number of complete zero attr= 81

Mean:
MEAN= 0.015643463154524837
Number of actScores= 1743
Number of complete zero attr= 83
Activation AVG per instance = 0.05647411375226131
STD = 1.001280445084084
Activation ABS sum = 0.9843438027019147

#### IG
Median:
MEDIAN= 0.0
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 49.940571350682724 %
STD = 0.5469943162127013 %
Activation ABS sum = 910.9160214364529

Mean:
MEAN= 0.004299697251521968
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 13.693348377279236
STD = 7.458846063627292
Activation ABS sum = 249.76667440157325

#### Lime
Median:
median= 0.052220337092876434
Number of actScores= 1807
Number of complete zero attr= 17
Activation AVG per instance = 64.26403987936276
STD = 15.452520319490699
Activation ABS sum = 1161.251200620085

Mean:
mean= 0.19865114641374293
Number of actScores= 1807
Number of complete zero attr= 17
Activation AVG per instance = 52.709100126197605
STD = 19.720295819054414
Activation ABS sum = 952.4534392803907




### CDEP (reg=10)
* IG: Number img: 1824, abs attr: 28.40, avg_attr_per_img: **0.01557** -> **1.5%**
* Gradcam: Number img: 1824, abs attr: 146.21, avg_attr_per_img: **0.0802** -> **8%**
* Lime:  Number img: 85, abs attr: 31.69, avg_attr_per_img: **0.3727** -> **37.27%**

#### Gradcam
Median:
Model ISIC19-CDEP-reg10-seed=100 loaded! Was trained on CDEPLoss
MEDIAN= 0.16095417737960815
Activation AVG per instance = 15.731144055916957 %
STD = 18.995598380684452 %
Activation ABS sum = 286.9360675799253
Number of complete zero attr= 0

Mean:
MEAN= 0.19656240007545994
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 14.608781655762446
STD = 18.303481401196255
Activation ABS sum = 266.464177401107



#### IG
Median:
MEDIAN= 3.97325748053845e-05
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 49.987748806087076 %
STD = 0.47001552698420335 %
Activation ABS sum = 911.7765382230282

Mean:
MEAN= 0.0157726926759974
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 29.900300492156756
STD = 6.517136031059495
Activation ABS sum = 545.3814809769392

#### Lime
Median:
median= 0.04313331097364426
Number of actScores= 1808
Number of complete zero attr= 16
Activation AVG per instance = 67.58878034135415
STD = 14.12126617398998
Activation ABS sum = 1222.005148571683

Mean:
mean= 0.22695530425402774
Number of actScores= 1808
Number of complete zero attr= 16
Activation AVG per instance = 56.99839504394192
STD = 16.23628069436734
Activation ABS sum = 1030.5309823944



### CE (all random)

#### Gradcam
Median:
median= 0.0
Number of actScores= 1824
Number of complete zero attr= 8
Activation AVG per instance = 13.234307541451448
STD = 15.554319192817312
Activation ABS sum = 241.3937695560744

Mean:
mean= 0.038638724760650736
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 11.893183828463231
STD = 13.593978602870859
Activation ABS sum = 216.93167303116934

#### IG
Median:
median= 2.4427772586932406e-05
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 49.88661925716881
STD = 0.5221259268420197
Activation ABS sum = 909.9319352507591

Mean:
mean= 0.0077691176153613225
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 29.873681603752182
STD = 8.716102004044656
Activation ABS sum = 544.8959524524398

#### Lime
Median:
median= 0.06620195508003235
Number of actScores= 1751
Number of complete zero attr= 73
Activation AVG per instance = 66.7302776184433
STD = 17.39804233542602
Activation ABS sum = 1168.4471610989422

Mean:
mean= 0.18626249472433537
Number of actScores= 1751
Number of complete zero attr= 73
Activation AVG per instance = 55.48619736162313
STD = 20.743643964972673
Activation ABS sum = 971.563315802021
Number of complete zero attr= 73


### HINT (reg=1)
#### Gradcam
Median:
median= 0.0
Number of actScores= 1159
Number of complete zero attr= 665
Activation AVG per instance = 0.8849328216908344
STD = 4.755208738017545
Activation ABS sum = 10.25637140339677

Mean:
mean= 0.009465588569099516
Number of actScores= 1159
Number of complete zero attr= 665
Activation AVG per instance = 0.8323845351753446
STD = 4.381693550325135
Activation ABS sum = 9.647336762682244

#### IG
Median:
median= 9.786320333660115e-06
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 50.04961373317137
STD = 0.4748024471326952
Activation ABS sum = 912.9049544930458

Mean:
mean= 0.007995489296598336
Number of actScores= 1824
Number of complete zero attr= 0
Number of actScores= 1824
Activation AVG per instance = 31.79277363151573
STD = 7.721947092935719
Activation ABS sum = 579.900191038847


#### Lime
Median:
median= 0.025923543609678745
Number of actScores= 1824
Activation AVG per instance = 64.40373560323853
STD = 14.10541669353574
Activation ABS sum = 1174.724137403071

Mean:
mean= 0.1533772428969346
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 52.95462713968989
STD = 16.61174839521896
Activation ABS sum = 965.8923990279436

# Interaction efficiancy

## RRRGradCAM NEXPL=1000 (most informative)
-> 1000 of 7386 feedback masks used

params: reg_rate=1, epochs= 50, batch_size=16, lr=0.001, optim=SGD(momentum=0.9), seeds=100, scheduler=True, shuffle=True, rr_clipping=1.0
**Train acc: 100.0%**, **train loss: 0.000002**

### Test set patches
Acc: 91.7%, Loss: 0.050571

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1824       | **0**  |
| cancer     | 193        | 299    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.90      | 1.00   | 0.95     | 1824    |
| cancer      |   1.00    |**0.61**| 0.76     | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.90     | 2316    |
| macro avg   | 0.95      |**0.80**| **0.85** | 2316    |
| weighted avg| 0.92      | 0.92   | 0.91     | 2316    |


### Test set no patches
Acc: 86.5%, Avg loss: 0.066601 

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1980       | **162**|
| cancer     | 193        | 299    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.91      |  0.92  | 0.92     | 2142    |
| cancer      | **0.65**  | 0.61   | **0.63** | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.86     | 2634    |
| macro avg   | 0.78      | 0.77   | 0.77     | 2634    |
| weighted avg| 0.86      | 0.87   | 0.86     | 2634    |


### Quantification WR
rrrg:
IG:
mean= 0.007234585573091304
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 25.87257732245091
STD = 8.52110844693178

GradCAM:
mean= 0.012405918869799097
Number of actScores= 1804
Number of complete zero attr= 20
Activation AVG per instance = 0.20613831549414
STD = 2.8572737532303396
Activation ABS sum = 3.7187352115142858

LIME:
mean= 0.19647740724773435
Number of actScores= 1800
Number of complete zero attr= 24
Activation AVG per instance = 57.08561496343464
STD = 19.225783729306787
Activation ABS sum = 1027.5410693418235


## RRR nexpl=1000

### Quantification WR
GradCAM:
mean= 0.39143380378127884
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 27.721894008870322
STD = 20.920371923774827
Activation ABS sum = 505.64734672179475

IG:
mean= 0.00981009720909928
Number of actScores= 1824
Number of complete zero attr= 0
Number of actScores= 1824
Activation AVG per instance = 7.203857642377698
STD = 8.38557841837153
Activation ABS sum = 131.39836339696922
Number of complete zero attr= 0

LIME:
mean= 0.19475018815545922
Number of actScores= 1824
Number of complete zero attr= 0
Activation AVG per instance = 46.41977136169245
STD = 23.190376262064813
Activation ABS sum = 846.6966296372702





# Reward vs Penalize experiment
This experiment investigates the difference between rewarding right reasons vs penalizing wrong reasons more closely. It's core question is 'which strategy leads to better results'. Note that it must be defined what exactly 'right reason' is beforehand. In many classification tasks, we can define the region of the object to be classified in the image as a relatively robust aproximation for right reasons. Although we can not clearly define what specific pixels in the target object are most crucial for the determination of the class with this approach - for humans this task is also ill-defined and not straightforward - we can still use this approach to exclude all information of the background, which is clearly wrong reasons. 

We extracted the provided segmentation masks from the ISIC 2019 dataset, which indicate the region of the skin lesions which should be classifed. This region corresponds to the 'right' reason, which is rewarded by the XIL HINT method. Inverting those masks gives us the wrong reasons (backgrounds), which are penalized via XIL RRRGradCAM. From the 17829 training instances, 11010 instances (62%) contain such a segmentation mask. In contrast to the experiments before the segmentation masks do not specifically indicate the confounding factor - the patches. Furthermore, there are right reasons masks available for almost all images with patches plus more for cancerous/not cancerous images without patches.
  
| stat \ model        | Vanilla          | HINT  (Reward)   | RRR-G (Penalize)|
|---------------------|----------------- |------------------|----------------|
| GradCAM all zero attr| 0\%             | 0.09\%           |  98.6\%        |
| GradCAM (Median) WR | 20.03% (±16.12)  | 1.83%            |  **6.63%         |
| GradCAM (Mean)   WR | 18.3% (±14.74)   | 1.71%            |  **6.20%         |
| IG (Mean)   WR      | 32.5%            | 34.3\%           |  26.4\%        |
| LIME (Mean)         | 53.0             | 58.0\%           |               |




## Reward right reason (HINT)
params: reg_rate=1; epochs= 50, batch_size=16, lr=0.001, optim=SGD(momentum=0.9), seeds=100, scheduler=True, train_shuffle=True  

**Train acc: 99.5%**, **train loss: 0.001977**

### Test set patches
Acc: 91.8%, Loss: 0.030890

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1824       | **0**  |
| cancer     | 191        | 301    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.91      |1.00    | 0.95     | 1824    |
| cancer      | 1.00      |0.61    | 0.76     | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.92     | 2316    |
| macro avg   | 0.95      | 0.81   | 0.85     | 2316    |
| weighted avg| 0.93      | 0.92   | 0.91     | 2316    |


### Test set no patches
Acc: 87.5%, Avg loss: 0.040490 

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 2005       | 137    |
| cancer     | 191        | 301    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.91      | 0.94   | 0.92     | 2142    |
| cancer      | 0.69      | 0.61   | 0.65     | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.88     | 2634    |
| macro avg   | 0.80      | 0.77   | 0.79     | 2634    |
| weighted avg| 0.87      | 0.88   | 0.87     | 2634    |

### Interpretation
* performance on both test sets are nearly identical to the vanilla model; no improvement on the test set without patches; Also no decrease on the test set with patches.

### Heatmaps analysis
#### Quantification of wrong reason in patches region:
* Median Binarization -> 1.83%
* Mean Binarization   -> 1.71%

GradCAM:
Additional info:
median= 0.0
Number of actScores= 1657
Number of complete zero attr= 167
Activation AVG per instance = 1.8265784865371826
STD = 7.746776244303601
Activation ABS sum = 30.266405521921115

mean= 0.018796217587214992
Activation AVG per instance = 1.7123659264468984
STD = 7.317071479581694
Activation ABS sum = 28.373903401225107

LIME:
mean= 0.13146902509380884
Number of actScores= 1824
Number of complete zero attr= 0
Number of actScores= 1824
Activation AVG per instance = 58.00573985100511
STD = 15.301190175261233
Activation ABS sum = 1058.0246948823333

IG:
mean= 0.007744038032174821
Number of actScores= 1824
Number of complete zero attr= 0
Number of actScores= 1824
Activation AVG per instance = 34.307356551698035
STD = 6.551746767715196
Activation ABS sum = 625.7661835029721


#### Quantification of right reason
todo

ISIC19-HINT-RPEXP-reg=1-seed=100.pt


## Penalize wrong reason (RRRGradCAM)
note: hard to train, because a huge region in the image is getting penalized --> leads to big rr loss and therefore unstable loss updates per batches (probably fixes with rr clipping, loss clipping, small reg rate)

params: reg_rate=0.1; rr_clipping=1.0, reduction='mean' epochs= 50, batch_size=16, lr=0.001, optim=SGD(momentum=0.9), seeds=100, scheduler=True, train_shuffle=True  

**Train acc: 100.0%**, **train loss: 0.000034** ra_loss = 0.000003, rr_loss=0.000031

### Test set patches
Acc: 91.8%, Loss: 0.048542

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1824       | **0**  |
| cancer     | 190        | 302    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.91      |1.00    | 0.95     | 1824    |
| cancer      | 1.00      |0.61    | 0.76     | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.91     | 2316    |
| macro avg   | 0.95      | 0.81   | 0.86     | 2316    |
| weighted avg| 0.93      | 0.92   | 0.91     | 2316    |


### Test set no patches
Acc: 87.1%, Avg loss: 0.063965 

**Confusion matrix:** 

| true \ pred| not_cancer | cancer |
|------------|------------|--------|
| not_cancer | 1991       | 151    |
| cancer     | 190        | 302    |  

**Classification report:**  

|             | precision | recall | f1-score | support |
|------------ |-----------|--------|----------|---------|
| not_cancer  | 0.91      | 0.93   | 0.92     | 2142    |
| cancer      | 0.67      | 0.61   | 0.64     | 492     |
|             |           |        |          |         |
| acc         |           |        | 0.86     | 2634    |
| macro avg   | 0.79      | 0.77   | 0.78     | 2634    |
| weighted avg| 0.87      | 0.87   | 0.87     | 2634    |

### Interpretation


### Heatmaps analysis
* nearly zero heatmaps could be generated for the test images with patches (only 25 of 1824). These heatmaps were all zeros, meaning gradcam cannot attribute any region in the images that positively influence the prediction.
* we also tried to generate gradcam heatmaps on the test set with no patches -> 798 of 
* NEED TO GENERATE IG and LIME HEATMAPS
     
#### Quantification of wrong reason in patches region:
* Median Binarization -> 6.63%
* Mean Binarization   -> 6.20%

GradCAM:
Additional info:
median= 0.0
Number of actScores= 25
Number of complete zero attr= 1799
Activation AVG per instance = 6.635572399944067
STD = 19.238894856439597
Activation ABS sum = 1.6588930999860168

mean= 0.009995774598792195
Activation AVG per instance = 6.201182898133993
STD = 17.59519421275091
Activation ABS sum = 1.5502957245334983


IG:
mean= 0.004826042911525674
Number of actScores= 1824
Number of complete zero attr= 0
Number of actScores= 1824
Activation AVG per instance = 26.39014714993863
STD = 8.279692590883968
Activation ABS sum = 481.35628401488066

LIME:
0.19420728464725556
Number of actScores= 1779
Number of complete zero attr= 45
Number of actScores= 1779
Activation AVG per instance = 58.16355875142006
STD = 19.7703955779495
Activation ABS sum = 1034.729710187763

#### Quantification of right reason
todo

<-->







IE-ISIC19-RRRGradCAM-nexpl=1000-reg=1-seed=100-clip=1.pt