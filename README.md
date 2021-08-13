# DomAd
Our codes are developed based on the codes released for CDAN. (Long M, Cao Z, Wang J, et al. Conditional adversarial domain adaptation[C]//Advances in Neural Information Processing Systems. 2018: 1640-1650.)

## Dataset
### Office-31
Office-31 dataset can be found on https://people.eecs.berkeley.edu/~jhoffman/domainadapt/. 
After downloading the data set, put the image file under the path "../data/office/domain_adaptation_images/".
Data list files named with format "*_10_list.txt" should be used as source data. "*_11_list.txt" should be used as target data.

### Office-Home
Office-Home dataset can be found on http://hemanthdv.org/OfficeHome-Dataset/.
After downloading the data set, put the image file under the path "../data/office-home/images/".
Data list files named with format "*_k.txt" should be used as source data. "*_uk.txt" should be used as target data.

## Training
Here we give an example of the training command for task D->A:

python train_image.py --gpu_id 0 --net ResNet50 --dset office --test_interval 50 --s_dset_path ../data/office/dslr_10_list.txt --t_dset_path ../data/office/amazon_11_list.txt OSDA

This command can be run directly after decompression.



