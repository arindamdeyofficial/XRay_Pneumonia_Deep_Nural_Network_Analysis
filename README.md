# XRay_Pneumonia_Deep_Nural_Network_Analysis
Predict Pneumonia from XRay using Deep Nural Network

Introduction of the problem: 
What is Pneumonia?
Pneumonia is a form of acute respiratory infection that affects the lungs. The lungs are made up of small sacs called alveoli, which fill with air when a healthy person breathes. When an individual has pneumonia, the alveoli are filled with pus and fluid, which makes breathing painful and limits oxygen intake.

Pneumonia is the single largest infectious cause of death in children worldwide. Pneumonia killed 740 180 children under the age of 5 in 2019, accounting for 14% of all deaths of children under five years old but 22% of all deaths in children aged 1 to 5. Pneumonia affects children and families everywhere, but deaths are highest in South Asia and sub-Saharan Africa (World Health Organisation, 2021).

https://www.svhlunghealth.com.au/Images/UserUploadedImages/3447/4_SVH_Lung_Health_Pneumonia_final_1080p.jpg

The Importance of Diagnosing Pneumonia?
The risk of pneumonia is immense for many, especially in developing nations where billions face energy poverty and rely on polluting forms of energy. Over 150 million people get infected with pneumonia on an annual basis especially children under 5 years old. In such regions, the problem can be further aggravated due to the dearth of medical resources and personnel. For example, in Africa’s 57 nations, a gap of 2.3 million doctors and nurses exists. For these populations, accurate and fast diagnosis means everything. It can guarantee timely access to treatment and save much needed time and money for those already experiencing poverty (Stephen, Sain, Maduh, & Jeong, 2019).

World Health Organisation. (2021, 11th of November). Pneumonia. Consulted on 13th of December 2021, van https://www.who.int/news-room/fact-sheets/detail/pneumonia
Stephen, O., Sain, M., Maduh, U. J., & Jeong, D. U. (2019). An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare. Journal of Healthcare Engineering, 2019, 1–7. https://doi.org/10.1155/2019/4180949
The Data: X-Ray Images
A total of 5,856 X-ray images of anterior-posterior chests were carefully chosen from retrospective pediatric patients between 1 and 5 years old. The dataset contains two kinds of chest X-ray Images: NORMAL and PNEUMONIA, which are stored in three folders. In the PNEUMONIA folder, two types of specifc PNEUMONIA can be recognized by the fle name: BACTERIA and VIRUS.
Experts versus AI
Despite the fact that pneumonia is the most common cause of serious illness and death in young children worldwide, our ability, as clinicians, to infer an infectious pathological process in the lung from specific features of the history and examination is poor (Scott et al., 2012).

Misdiagnosis, arbitrary charges, annoying queues, and clinic waiting times among others are long-standing phenomena in the medical industry across the world. These factors can contribute to patient anxiety about misdiagnosis by clinicians. However, with the increasing growth in use of big data in biomedical and health care communities, the performance of artificial intelligence (Al) techniques of diagnosis is improving and can help avoid medical practice errors (Daniel et al., 2017).

Some research on medical image classification by CNN has achieved performances rivaling human experts. For example, CheXNet, a CNN with 121 layers trained on a dataset with more than 100,000 frontal-view chest X-rays (ChestX-ray 14), achieved a better performance than the average performance of four radiologists (Shen et al., 2019).

Most of the experts got high sensitivity but low specificity, while the CNN-based system got high values on both sensitivity and specificity (Nagendran et al., 20120). Moreover, on the average weight error measure, the CNN-based system exceeds two human experts (Liu et al., 2019).

The development of diverse AI techniques has contributed to early detections, disease diagnoses, and referral management. In addition, more than half of a randomized population group (55.8%: 428 out of 767) opted for AI diagnosis regardless of the description of the clinicians (Liu et al., 2021).

Daniel, P., Bewick, T., Welham, S., Mckeever, T. M., & Lim, W. S. (2017). Adults miscoded and misdiagnosed as having pneumonia: results from the British Thoracic Society pneumonia audit. Thorax, 72(4), 376–379. https://doi.org/10.1136/thoraxjnl-2016-209405
Liu, X., Faes, L., Kale, A. U., Wagner, S. K., Fu, D. J., Bruynseels, A., Mahendiran, T., Moraes, G., Shamdas, M., Kern, C., Ledsam, J. R., Schmid, M. K., Balaskas, K., Topol, E. J., Bachmann, L. M., Keane, P. A., & Denniston, A. K. (2019). A comparison of deep learning performance against health-care professionals in detecting diseases from medical imaging: a systematic review and meta-analysis. The Lancet Digital Health, 1(6), e271–e297. https://doi.org/10.1016/s2589-7500(19)30123-2
Liu, T., Tsang, W., Huang, F., Lau, O. Y., Chen, Y., Sheng, J., Guo, Y., Akinwunmi, B., Zhang, C. J., & Ming, W. K. (2021). Patients’ Preferences for Artificial Intelligence Applications Versus Clinicians in Disease Diagnosis During the SARS-CoV-2 Pandemic in China: Discrete Choice Experiment. Journal of Medical Internet Research, 23(2), e22841. https://doi.org/10.2196/22841
Nagendran, M., Chen, Y., Lovejoy, C. A., Gordon, A. C., Komorowski, M., Harvey, H., Topol, E. J., Ioannidis, J. P. A., Collins, G. S., & Maruthappu, M. (2020). Artificial intelligence versus clinicians: systematic review of design, reporting standards, and claims of deep learning studies. BMJ, m689. https://doi.org/10.1136/bmj.m689
_Scott, J. A. G., Wonodi, C., Moïsi, J. C., Deloria-Knoll, M., DeLuca, A. N., Karron, R. A., Bhat, N., Murdoch, D. R., Crawley, J., Levine, O. S., O’Brien, K. L., & Feikin, D. R. (2012). The Definition of Pneumonia, the Assessment of Severity, and Clinical Standardization in the Pneumonia Etiology Research for Child Health Study. Clinical Infectious Diseases, 54(suppl2), S109–S116. https://doi.org/10.1093/cid/cir1065
Shen, J., Zhang, C. J. P., Jiang, B., Chen, J., Song, J., Liu, Z., He, Z., Wong, S. Y., Fang, P. H., & Ming, W. K. (2019). Artificial Intelligence Versus Clinicians in Disease Diagnosis: Systematic Review. JMIR Medical Informatics, 7(3), e10010. https://doi.org/10.2196/10010
How do Deep Learning Networks distinguish between healthy and unhealthy lungs?
Most deep neural network applied to the task of pneumonia diagnosis have been adapted from natural image classification. Since natural image classification models have a large number of parameters as well as high hardware requirements, which makes them prone to overfitting and harder to deploy in mobile settings (Fourcade & Khonsari, 2019).

Convolutional Neural Networks are a common form of deep networks for classification tasks. CNNs have extensive learning capacity and can infer the nature of an input image without any prior knowledge, which makes them a suitable method for image classification (Toraman, Alakus, & Turkoglu, 2020). CNNs make use of the following three properties:

1. First: units in each layer receive inputs from the previous units which are located in a small neighborhood. This way, elementary features such as edges and corners can be extracted. Then these features will be combined in next layers to detect higher order features.

2. Second: important property is the concept of shared weights, which means similar feature detectors are used for the entire image.

3. Third: CNNs usually have several sub-sampling layers. These layers are based on the fact that the precise location of the features are not only beneficial, but also harmful, because this information tends to vary for different instances (Yadav & Jadhav, 2019).

Fourcade, A., & Khonsari, R. (2019). Deep learning in medical image analysis: A third eye for doctors. Journal of Stomatology, Oral and Maxillofacial Surgery, 120(4), 279–288. https://doi.org/10.1016/j.jormas.2019.06.002
Yadav, S. S., & Jadhav, S. M. (2019). Deep convolutional neural network based medical image classification for disease diagnosis. Journal of Big Data, 6(1). https://doi.org/10.1186/s40537-019-0276-2
Stephen, O., Sain, M., Maduh, U. J., & Jeong, D. U. (2019). An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare. Journal of Healthcare Engineering, 2019, 1–7. https://doi.org/10.1155/2019/4180949
Toraman, S., Alakus, T. B., & Turkoglu, I. (2020). Convolutional capsnet: A novel artificial neural network approach to detect COVID-19 disease from X-ray images using capsule networks. Chaos, Solitons & Fractals, 140, 110122. https://doi.org/10.1016/j.chaos.2020.110122
Our CNN Model: 
The Architecture
Our architecture for the CNN has been inspired by the article from Stephan and colleagues (2019) and Yadav and Sjadav (2019). Their neural network architectures were specifically designed for pneumonia image classification tasks. The proposed architecture consists of the convolution, max-pooling, and classification layers combined together. We will now dive into each component and why we chose them.

https://miro.medium.com/max/2656/1*18A5bLKeQKCRuOIT17pT8A.png

The Convolution
The main building block of CNN is the convolutional layer. Convolution is a mathematical operation to merge two sets of information. In our case the convolution is applied on the input data using a convolution filter to produce a feature map.

https://cdn-images-1.medium.com/max/800/1*VVvdh-BUKFh2pwDD0kPeRA@2x.gif

Hyperparameters: 
Pooling
We performed pooling to reduce the dimensionality. This enables us to reduce the number of parameters, which both shortens the training time and combats overfitting. Pooling layers downsample each feature map independently, reducing the height and width, keeping the depth intact. We used Max Pooling by taking the max value in the pooling window. We saw it as important for downsampling the feature map while keeping the important information. The max-pooling layer of the convolutional neural network is effective in variant shape absorptions and comprises sparse connections in conjunction with tied weights.

https://miro.medium.com/max/4800/1*ReZNSf_Yr7Q1nqegGirsMQ@2x.png

Fully Connected
After the convolution and pooling layers we add a couple of fully connected layers to wrap up the CNN architecture.

Improve performance by preventing overfitting: 
We used various various strategies to increase the performance of image classifcation by preventing overfitting:

"Only a network model with proper size and other effective methods preventing overfit, such as proper dropout rate and proper data augmentation, can get the best results." (Yadav & Jadhav, 2019)

Data Augmentation
We employed several data augmentation methods to artificially increase the size and quality of the dataset. The idea is to alter the training data with small transformations to reproduce the variations. This process helps in solving overfitting problems and enhances the model’s generalization ability during training (Stephen, Sain, Maduh, & Jeong, 2019). That is because augmentation geometrically transforms the picture, which facilitates the machine learning algorithm to learn the underground feature without the impact of rotation and scale (Yadav & Jadhav, 2019).

In addition, data augmentation can be much more helpful when the dataset is imbalanced (yadav & Jadhav, 2019) which is the case here. We can generate different samples of the undersampled class in order to try to balance the overall distribution.

Dropout
Dropout is used to prevent overfitting by temporarily “dropping” a neuron during training time at each iteration with probability p. Which means that all the inputs and outputs to this neuron will be disabled at the current iteration. The dropped-out neurons are resampled with probability p at every training step, so a dropped out neuron at one step can be active at the next one. The hyperparameter p is called the dropout-rate and we set it to 0.5, corresponding to 50% of the neurons being dropped out which is proposed as the best option for X-ray image classification (yadav & Jadhav, 2019).

It has to search for broad, general patterns, whose weight patterns tend to be more robust. It was found to be of high importance when it comes to classification of X-ray images (Stephan et al., 2019).

https://miro.medium.com/max/4800/1*7LrJUUXIO8ewrbuUIbUkXQ@2x.png

Batch Normalization
Use batch norm with convolutions. As the network becomes deeper, batch normalization start to play an important role, to speed up the training procedure and reduce overfitting. The idea is to alter the training data with small transformations to reproduce the variations, which makes learning more stable and quicker (Yadav & Jadhav, 2019).

Regularization: 
In order to prevent our model to overtrain we implement the following regularization measures: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint which was found to be very effective for X-ray image classification (Singh, Kumar, Yadav, & Kaur, 2020).

The early stopping callback stops the training process when the model starts becoming stagnant, or even worse, when the model starts overfitting.
We have adopted "ReduceLROnPlateau" as a Keras callback function to reduce the learning rate when the result stops improving. The learning rate will gradually reduce by a factor of 0.3. This function also helps the network to reduce the overfitting problem.
The checkpoint callback saves the best weights of the model, so next time we want to use the model, we do not have to spend time training it.
Training
CNN is trained using backpropagation with gradient descent.

Yadav, S. S., & Jadhav, S. M. (2019). Deep convolutional neural network based medical image classification for disease diagnosis. Journal of Big Data, 6(1). https://doi.org/10.1186/s40537-019-0276-2
Singh, D., Kumar, V., Yadav, V., & Kaur, M. (2020). Deep Neural Network-Based Screening Model for COVID-19-Infected Patients Using Chest X-Ray Images. International Journal of Pattern Recognition and Artificial Intelligence, 35(03), 2151004. https://doi.org/10.1142/s0218001421510046
Stephen, O., Sain, M., Maduh, U. J., & Jeong, D. U. (2019). An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare. Journal of Healthcare Engineering, 2019, 1–7. https://doi.org/10.1155/2019/4180949
Now: The Challenge!! 
Build an algorithm to automatically identify whether a patient is suffering from pneumonia or not by looking at chest X-ray images. The algorithm had to be extremely accurate because lives of people is at stake.

Load the data and all the necessary packages
We prepare the data which we use in our Convolutional Neural Network. The Chest X-ray data is given into three saperate folders: train, val, and test. Run following cell to set dataset path and other few variables which are used by ImageDataGenerator in next step.

Our data is located in three folders:¶
Train: the folder that contains the training images for training our model.
Val: the folder that contains images which we will use to validate our model. A validation dataset is has its purpose to prevent our model from Overfitting. Overfitting is when the loss is not as low as it could be because the model learned too much noise. Therefore it can't handle data it hasn't see too well.
Test: this folder contains the data that we use to test the model once it has learned the relationships between the images and their label (Pneumonia versus Not-Pneumonia)
