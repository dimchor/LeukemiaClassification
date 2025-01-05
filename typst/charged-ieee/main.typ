#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [
    Classification of Leukaemic Cancer Cells from Normal Cells Using a
    Convolutional Neural Network
  ],
  abstract: [
    Acute Lymphoblastic Leukaemia (ALL) is a type of malignant cancer of the
    lymphoid line of blood cells. ALL most commonly occurs in children and it
    has generally good prognosis. It is vital that ALL be detected early
    and treated promptly. Therefore, a system capable of diagnosing potential
    patients by analysing images of their blood smear may facilitate in earlier
    diagnosis. In this report, it is proposed that a Convolutional Neural
    Network (CNN) be used to classify malignant cells (B-ALL) from normal cells.
    The method presented in this paper yields mediocre results; the accuracy of
    the model in question does not exceed 68%.
  ],
  authors: (
    (
      name: "Dimitrios Chorevas",
      department: [Department of Informatics and Computer Engineering],
      organization: [University of West Attica],
      location: [Aegaleo, Attica, Greece],
      email: "dim.chorevas@gmail.com"
    ),
  ),
  index-terms: (),
  bibliography: bibliography("refs.yml"),
  figure-supplement: [Fig.],
)

= Introduction
Acute Lymphoblastic Leukaemia (ALL) is a malignancy of blood cells. It is
characterised by uncontrollable hyperplasia of lymphoblasts in the bone
marrow with little to no cell differentiation. Lymphoblasts then migrate to
other organs of the body, through the bloodstream, causing tissue death.

ALL is the most common type of leukaemia in children; comprising 80% of total
leukaemia cases. Children with Down syndrome or other genetic disorders are more
likely to develop ALL. Adults over the age of 60 may also develop ALL due to
chromosomal and molecular discorders.

There are many variants of ALL. All of which have one common characteristic,
namely the presence of lymphoblasts in the bone marrow and blood. There are
three ALL groups, specifically:
- B-lymphoblastic leukemia/lymphoma without specific genetic characteristics,
- B-lymphoblastic leukemia/lymphoma with specigic genetic disorders and
- T-lymphoblastic leukemia/lymphoma.
@haematology-lessons

Symptoms of ALL are mainly caused by cytopenias (anaemia, thrombocytopenia,
leukopenia and/or neutropenia). Some of these include, but not limited to:
- fever (60%),
- pallor (40%),
- excesive bleeding (50%) due to thrombocytopenia,
- ostalgia (bone pain) (25%),
- lymphadenopathy (50%) and
- hepatosplenomegaly (70%) among others.
@practical-guide

= Related Work

Researchers have employed a wide variety of machine learning techniques to
approach this problem @classification-knn-ieee @classification-cnn-elsevier
@classification-cnn-springer @classification-cnn-x-ray. A common pattern
observed in many studies is the use of image processing techniques to enhance
important features of input images. In particular, cell segmentaion has a vital
role in the appropriate classification of input images
@classification-cnn-elsevier. Other techniques, such as normalisation, contrast
enhancement and noise removal have been used to improve the overall quality of
input images. As for the classification algorithm, K-means clustering has been
used widely to classify blood cells, showing a significant accuracy rate of 98%
@classification-knn-ieee. Convolutional Neural Networks (CNNs) are increasingly
getting more popular as a means of classifying images. Researchers have used
models such as Mobilenet, ResNet, AlexNet, DenseNet and VGG16 with accuracy
rates reaching up to 99.39% @classification-cnn-elsevier
@classification-cnn-springer.

= Proposed Method

In this report, a CNN is used to extract features from the C-NMC dataset
@ds-self @ds-cite-1 @ds-cite-2 @ds-cite-3 @ds-cite-4 @ds-cite-5 and
perform classification. The dataset contains *15.135* images from *118*
patients. There are two classes: normal and leukaemia blast.
The images have already been processed by the authors of the dataset. The
authors performed cell segmentation and noise reduction from the microscopic
images using their "own in-house method of stain color normalisation" @ds-self.

Further image processing is conducted to input data. While the initial images
use the RGB colour profile, the processed images use the grayscale profile
instead. Additionally, the images are resize to 128 by 128 pixels. By reducing
the amount of data needed to be processed, the model becomes more performant.
This is crucial as the hardware resources used to perform the experiments were
limited. Next, histogram equalisation is used to increase the contrast of the
images in order to enhance its features. Finally, the image in converted into an
array and then normalised to further reduce numeric complexity.

#figure(
  caption: [Dataset image: UID_H12_24_5_hem.bmp],
  image("images/UID_H12_24_5_hem.png", width: 40%),
)

#figure(
  caption: [Processed image used by model],
  image("images/UID_H12_24_5_hem_processed.png", width: 40%),
)

Only a part of the dataset is used in this project. Specifically,
- 4.800 images are used for training the model
- 1.200 images are used as validation set and
- 1.200 images are used for testing the model.

The images in all three sets are arbitrarily chosen. The images used for
training are located in the training subdirectory. The images used for
validating and testing are located in the validation subdirectory. Even though
the dataset does have a set of images for the purpose of testing, the class
labels are not available publicly @ds-self. For this reason, the author decided
to pull additional images from the validation subdirectory to populate the
testing set.
