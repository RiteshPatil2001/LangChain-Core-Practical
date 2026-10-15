from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Please generate summarized notes for the following chapter: {chapter}.",
    input_variables=["chapter"]
)

prompt2 = PromptTemplate(
    template = "Please generate 5 mcq quiz questions on the following chapter: {chapter}.",
    input_variables=["chapter"]
)

parallel_chain = RunnableParallel({
    'summarized_notes': prompt1 | model | parser,
    'mcq_quiz_questions': prompt2 | model | parser
})

prompt3 = PromptTemplate(
    template = "Please merge summarized notes and mcq quiz questions into a single report: {summarized_notes} {mcq_quiz_questions}.",
    input_variables=["summarized_notes", "mcq_quiz_questions"]
)

# merging_chain = parallel_chain | prompt3 | model | parser
merging_chain = prompt3 | model | parser

final_chain = parallel_chain | merging_chain

text = """
Machine learning algorithms are sets of rules that allow computers to learn from data, identify patterns and make predictions without being explicitly programmed. Instead of following fixed instructions, these algorithms improve their performance as they are exposed to more data. Machine learning is commonly categorized into three types:

Supervised Learning: Algorithms learn from labeled data, where the input-output relationship is known.
Unsupervised Learning: Algorithms work with unlabeled data to identify patterns or groupings.
Reinforcement Learning: Algorithms learn by interacting with an environment and receiving feedback in the form of rewards or penalties.
types_of_ml
Machine Learning Algorithms
Supervised Learning Algorithms
Supervised learning Algorithms are trained on datasets where each example is paired with a target or response variable, known as the label. The goal is to learn a mapping function from input data to the corresponding output labels, enabling the model to make accurate predictions on unseen data.

supervised_learning
Supervised Learning Algorithms
Supervised learning problems are mainly divided into:


Classification: predicting categories
Regression: predicting continuous values
1. Linear Regression
Linear regression predicts a continuous value by fitting a straight line between input and output variables. Example: Predicting house prices based on area or predicting weight from height.

2. Logistic Regression
Logistic regression is used for classification problems, not regression. Example: Spam detection (spam/not spam), disease prediction (yes/no)

Note: Despite the name, logistic regression is a classification algorithm.

3. Decision Trees
Decision trees split data into branches based on feature values. Popular Decision Tree Algorithms: ID3, C5, CART

4. Support Vector Machines (SVM)
SVMs find the best boundary (hyperplane) that separates data points into different classes. Uses support vectors to define decision boundaries.

5. k-Nearest Neighbors (k-NN)
k-NN predicts outcomes based on the closest data points in the training set. Uses distance metrics like Euclidean or Manhattan. Works for both classification and regression

6. Naive Bayes
Naive Bayes is based on Bayes’ Theorem and assumes features are independent. Common use case: Text classification, spam filtering

7. Random Forest
Random forest is an ensemble technique that combines multiple decision trees. It uses random sampling and feature selection. Reduces overfitting compared to a single tree.

8. Gradient Boosting
Gradient boosting builds models sequentially, where each model corrects errors made by the previous one. Combines weak learners into a strong model. It works for both classification and regression. Popular Variants:

XGBoost: faster, regularized boosting
LightGBM: histogram-based, fast for large datasets
CatBoost: handles categorical data efficiently
9. Neural Networks ( Including Multilayer Perceptron)
In supervised learning, Neural Networks learn patterns using layers of connected neurons. It require labeled data. It trained using backpropagation and Used for classification and regression

Multilayer Perceptron (MLP): Neural network with input, hidden and output layers
Applications: Image classification, spam detection, price prediction
Unsupervised Learning Algorithms
Unsupervised learning works with unlabeled data to discover hidden patterns or structures without predefined outputs. These are again divided into three main categories based on their purpose: Clustering, Association Rule Mining and Dimensionality Reduction.

unsupervised_learning
Unsupervised Learning Algorithms
1. Clustering
Clustering algorithms group similar data points into clusters to identify natural patterns or groupings. They are commonly divided into Centroid-based, Distribution-based, Connectivity-based and Density-based methods.

Centroid-Based Methods

K-Means: divides data into k clusters
K-Means++: Improves K-Means by choosing better initial cluster centroids.
K-Mode: Groups categorical data based on modes instead of means.
Fuzzy C-Means: Allows data points to belong to multiple clusters with different membership levels.
Distribution-Based Methods

Gaussian mixture models (GMMs): Represents data as a mixture of multiple Gaussian distributions.
Expectation-Maximization: Estimates model parameters when some data or variables are hidden.
Dirichlet Process Mixture Models (DPMMs): Automatically determines the number of clusters using a Bayesian approach.
Connectivity-Based Methods

Hierarchical clustering: Builds a hierarchy of clusters by repeatedly merging or splitting groups.
Agglomerative Clustering: Starts with individual points and progressively merges similar clusters.
Divisive clustering: Starts with one cluster and progressively splits it into smaller clusters.
Affinity propagation: Identifies representative data points called exemplars to form clusters.
Density-Based Methods
DBSCAN: detects arbitrary-shaped clusters and noise
OPTICS: Identifies clusters with different densities and detects noise in the data.
2. Dimensionality Reduction
Dimensionality reduction reduces the number of features while keeping important information.

PCA: maximizes variance
t-SNE: visualization of high-dimensional data
NMF: works with non-negative data
ICA: Separates data into statistically independent components.
Isomap: Reduces dimensions while preserving relationships between data points.
LLE: Preserves local relationships between neighboring data points while reducing dimensions.
LSA (for text data)
Autoencoders: learn compact representations of data by encoding and reconstructing the input.
3. Association Rule Mining
Association rule mining identifies relationships and patterns between items in large datasets, commonly used in market basket analysis. It finds frequent item combinations based on their occurrence and co-occurrence in the data.

Apriori: Finds frequent itemsets and generates association rules from them.
FP-Growth: Finds frequent itemsets efficiently using an FP-tree structure.
ECLAT: Finds frequent itemsets using a vertical representation of transaction data.
Reinforcement Learning Algorithms
Reinforcement learning trains an agent to make decisions by interacting with an environment.

Learns through rewards and penalties
Goal is to maximize long-term reward
agent
Reinforcement Learning Algorithms
1. Model-Based Methods
Value Iteration: Iteratively updates state values to find the optimal policy.
Monte Carlo Tree Search: Uses simulations to explore possible actions and select the most promising one.
2. Model-Free Methods
Value-Based Methods

Q-Learning: Learns the best action to take in each state by estimating future rewards.
SARSA: Learns action values based on the current state, action, reward and next state-action pair.
Monte Carlo Methods: Learn value functions by averaging rewards collected from complete episodes.
Policy-Based Methods

REINFORCE : Learns an optimal policy by updating action probabilities based on received rewards.
Actor-Critic : Combines a policy-based actor with a value-based critic to improve decision-making.
A3C: Uses multiple agents working in parallel to learn policies and value functions.
Other Machine Learning Algorithm
Semi-Supervised Learning Algorithms
Semi-supervised learning algorithms use both labeled and unlabeled data for training. These algorithms are useful when labeling data is expensive, but unlabeled data is easily available.

Self-Training: Uses a model's confident predictions to assign labels to unlabeled data.
Co-Training: Trains multiple models on different feature sets and uses their predictions to label unlabeled data.
Label Propagation: Assigns labels to unlabeled data based on nearby labeled data points.
Label Spreading: Propagates labels through a graph while allowing the labels to be adjusted during learning.
Example: Classifying images when only a small portion of images are labeled.

Self-Supervised Learning Algorithms
Self-supervised learning algorithms automatically generate labels from the data itself and learn meaningful representations without manual annotation.

Contrastive Learning (SimCLR, MoCo): Learns representations by bringing similar examples closer and separating dissimilar examples.
Masked Language Models (BERT-style training): Learn language representations by predicting masked or hidden words from their context.
Example: Predicting missing words in a sentence or hidden parts of an image.
"""

result = final_chain.invoke({'chapter': text})

print(result)

# final_chain.get_graph().print_ascii()
