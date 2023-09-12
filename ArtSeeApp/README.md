ArtSee Notebook Demo
----
![ArtSee Logo](https://raw.githubusercontent.com/ArtSee-App/Demo/main/Github%20Assets/artseelogo.jpg)

ArtSee is a Python application transforming the museum experience. By leveraging YOLOv8 for object detection and ResNet for feature extraction, it identifies paintings from your camera view and provides relevant information. The embeddings created by ResNet are matched with a vast database of paintings via a FAISS (Facebook AI Similarity Search) index. Google Cloud Platform (GCP) ensures efficient data management, enabling ArtSee to offer accurate and swift responses. Enjoy museums with ArtSee, your pocket-sized, personalized art curator.

![ArtSee Gif](https://raw.githubusercontent.com/ArtSee-App/Demo/main/Github%20Assets/artsee_if.gif)

To Get Started!
----
To get started with ArtSee, a sample dataset of images has been provided for demonstration purposes. Follow these simple steps to explore the capabilities of ArtSee:

1. Clone this repository to your local machine.
2. Navigate to the artsee-demo.ipynb file.
3. Run the Jupyter notebook and follow along with the comments provided in the code.
4. This will give you a hands-on understanding of how ArtSee works. Enjoy exploring art with the help of AI!

Feature Summary: Demonstration
----
ArtSee provides a comprehensive demonstration using a predefined dataset of images. Here's what you can expect:

* Painting Detection: The application will recognize and identify paintings in your image.
* Image Cropping: It isolates the identified paintings, cropping them from the surrounding image for further analysis.
* Embedding Generation: It will then create unique embeddings for these paintings, transforming the images into numerical representations that can be analyzed and compared.
* Nearest Neighbor Search: A search algorithm is then employed to find the most similar image in our dataset, using Euclidean distance as the measure of similarity.
In this context, the "closest" image in Euclidean space will correspond to the identical painting within the input image, given it's included in the dataset. Explore the power of AI in art recognition with ArtSee.

Background
----
TODO
