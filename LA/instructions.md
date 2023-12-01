#  Markov chains 
( this is for paper )

    A Markov chain or Markov process is a stochastic model describing a sequence of possible events 
    in which the probability of each event depends only on the state attained in the previous event. 
    Informally, this may be thought of as, "What happens next depends only on the state of affairs now.

    It has a wide variety of applications in all fields ranging from marketing, finance, Quantum-chemistry
    image processing, statistics, and last but not the least Biology.
    
    ## Population analysis

    ### Leslie Matrix
        In applied mathematics, the Leslie matrix is a discrete, age-structured model of 
        population growth that is very popular in population ecology. 
        The Leslie matrix (also called the Leslie model) is one of the most well-known 
        ways to describe the growth of populations (and their projected age distribution), 
        in which a population is closed to migration, growing in an unlimited environment, 
        and where only one sex, usually the female, is considered

        The below matrix L is called a Leslie Matrix, in general if we have a popluation with n classes
        of equal duration, L will be nxn matrix will the following strucgure:

        \usepackage{amsmath}

        \[
        L =
        \begin{pmatrix}
        b_1 & b_2 & 0 & \dots & b_n \\
        s_1 & 0 & 0 & \dots & 0 \\
        0 & s_2 & 0 & \dots & 0 \\
        0 & 0 & s_3 & \dots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 & \dots & s_{n-1} & 0 \\
        \end{pmatrix}
        \]    

        Here b1, b2, ... are the birth parameters ( b_i = the average numbers of female produced by each female in 
        class i) and s_1, s_2, ... are the survival probability ( s_i = the probability that a female in class i
        survives into class i + 1 ).

        Multiplying a matrix with its state vector gives the matrix in the next state just like a markov chain.
        In this way we can predict one or more steady states where this model can reach.

    ### Prey-predator model

        Prey-Predator (also known in literature as Lotka-Volterra model) is a popular model to study dynamics of a system consisting of two antogonists, in this case rabbits (prey) and foxes (predator). 

        The dynamics of the sytem are determined by interactions within and between the prey and predator populations. The intra-species interactions are (natural) birth and (natural) death rates, while inter-species interactions are the predation of prey (i.e. predator 'eats' prey for its survival!). Let $X$ denote the population size of prey and $Y$ denote the popluation size of predator. 


        For the population  dynamics of the prey: prey replicates at a rate that is controlled by abundance of the natural resources (rabbits need grass); we assume that these natural resources are abundant and remain at the same level throughout. Prey might die of natural causes (old age) or is eaten by predator. Thus the dynamics are reasonably modeled as:
        $$ \frac{dX}{dt} = \alpha X - \beta X Y $$
        For the population dynamics of predator: population of predator is expected increase linearly with its own size, and also on the population size of prey (since it needs prey as food). The natural death rate of the population depends on its own population size. Thus prey population size dynamics may be modeled as:
        $$ \frac{dY}{dt} = \gamma X Y - \delta Y $$
        Clearly the dynamics of the model are dependent on the four positive constants $\alpha,~\beta,~\gamma$, and $\delta$, which are to be inferred from the feild data.

        But this model also has its limitations:
        - Linearity assumption: Linear algebra is based on the assumption of linearity, which may not hold true in all ecological systems. 
        - Lack of spatial considerations: Linear algebra treats populations as homogeneous entities without considering spatial heterogeneity.
        - Simplified assumptions: Linear algebra models often rely on simplifying assumptions to facilitate mathematical tractability.
        - Lack of feedback mechanisms: Prey-predator interactions involve feedback mechanisms, where changes in prey and predator populations influence each other

        The modern side of the research in the field of Lotka-Volterra models is under "Extinction Dynamics"
        and "Evolving Networks" \cite{logistic} which is way beyond the scope of our project

        
    ## CpG Islands

        Before we jump into CPG islands we need to know a few things:

        ### DNA Methylation -
            DNA Methylation also called Dmeth in short is an epigenetic modification of DNA that happens in all cells of your body. 
            DMeth levels across the genome have been successfully used to predict all kinds of diseases from 
            [cancers](\cite{cancer})
            to [Alzheimer's](\cite{alzheimer}), 
            your [biological age](\cite{Bell2019}) 
            and even your [time to death!](\cite{death}).
            The Problem Unfortunately, we still don't know what causes the changes in DMeth and which
            part effects what.    
            Thus we engineer smarter features using DNA methylation data and subsequently use 
            them in predicting cancer status. ( more about that later )
            In particular, we are looking for genomic features that underlie DNA Methylation 
            states at different DNA sites. 
        
        ### HMM ( Hidden Markov Chains ) -

            Hidden Markov Model (HMM) is a statistical model that is used to describe the probabilistic relationship 
            between a sequence of observations and a sequence of hidden states. 
            It is often used in situations where the underlying system or process that generates 
            the observations is unknown or hidden, hence it got the name “Hidden Markov Model.” 

            It is used to predict future observations or classify sequences, 
            based on the underlying hidden process that generates the data.

            An HMM consists of two types of variables: hidden states and observations.

            > The hidden states are the underlying variables that generate the observed data, 
            but they are not directly observable.
            > The observations are the variables that are measured and observed. 

            ( Above is just a brief introduction of HMMs )
            More in depth informaiton of Hidden Markov Chains is available in these

            \usepackage{hyperref}

            \newcommand{\multiplehyperlinks}[2]{%
                \foreach \link in {#1} {%
                    \href{\link}{#2}%
                }%
            }

            \begin{document}

            \multiplehyperlinks{%
                {https://www.example.com},
                {https://www.google.com},
                {https://www.wikipedia.org}%
            }{links}

        So coming back to our main topic...

        CpG islands (or CG islands) are regions with a high frequency of CpG sites.
        Though objective definitions for CpG islands are limited, the usual formal
        definition is a region with at least 200 bp (bond pairs), a GC percentage greater than 50%,
        and an observed-to-expected CpG ratio greater than 60%. 

        So we will move forward with this topic taking an example and replicating
        DMeth. 

    Glossary:

    Logistic Regression: 
        Just like Linear Regression, Logistic Regression is a statistical model which is used 
        to get the most accurate  estimation of the dataframe. But unlike Linear the values are binary
        in the sense, there are only two states a dependent variable can Exist in. Like mean squared error
        in Linear Regression, we try to maximize something called AUC ( Area under curve ) to measure accuracy



# THE CpG Island Model demonstration
    
    We have simlulated a similar project ( refer to the jupyter notebook attached ) which
    performs Logistic Regression to analyse the DMeth

    The goal of this section of the project is to predict DNA methylation state (Beta 0= unmethylated, Beta 1=methylated), 
    at any given CpG site. DNA samples are collected from more than 400 individuals and averaged 
    to calculated the methylation state at each CpG site.

    ## Methodology

    Here's a breakdown of the steps involved:

    - Data Loading: The code starts by loading the training data from the $"train.csv"$ 
    file into a pandas DataFrame using the $pd.read_csv()$ function.

    Data Preprocessing: The code then performs some preprocessing steps on the data. 
    First, it selects the relevant features for analysis, including the 
    "Relation_to_UCSC_CpG_Island" feature and the "Beta" values. 
    pd.get_dummies() function to convert categorical values into numerical values. 
    The "Beta" values are appended to the feature matrix.

    - Feature Analysis: The code calculates the mean of the "Beta" values for each unique 
    value in the "Island" feature. This provides an overview of the relationship 
    between CpG islands and the "Beta" values.

    - Data Split and Model Training: The feature matrix X is separated from the target variable y, 
    which contains the "Beta" values. 
    The logistic regression model is then trained on the training data using 
    the LogisticRegression class from the scikit-learn library.

    Model Evaluation: The code calculates the Area Under the ROC Curve (AUC) using the roc_auc_score() 
    function from scikit-learn. AUC is a commonly used metric for 
    evaluating binary classification models. Additionally, the code generates a 
    classification report using the classification_report() function to provide 
    detailed performance metrics such as precision, recall, and F1-score.

    Test Data Prediction: The code loads the test data from the "test.csv" file and preprocesses 
    it in the same way as the training data. The logistic regression model is then used to predict 
    the "Beta" values for the test data.

    Result Export: The predicted "Beta" values are saved in the "solution.csv" file along with the other 
    columns from the test data.

    ## Interpretation:

    By calculating the mean "Beta" value for each unique value in the "Island" feature, 
    the code provides an overview of the average methylation levels associated with different 
    types of CpG islands. This analysis implies that different CpG island types may 
    exhibit varying levels of methylation. Further exploration and analysis of 
    these associations can provide insights into the relationship between 
    CpG islands and methylation levels.

    The logistic regression model trained on the provided training data is used to predict the 
    methylation levels ("Beta" values) for the test data    
    This suggests that the model has learned patterns from the training data and can 
    generalize its predictions to unseen data. However, the specific accuracy or quality 
    of these predictions is not mentioned in the code.

    Overall, this code provides a simple approach to analyze CpG islands using logistic regression. 
    It leverages the relation of CpG islands to specific locations and associated methylation levels 
    to make predictions. The interpretation of the results and the insights gained depend on the 
    specific data and context in which the analysis is applied.



# Logistic Maps

    Logistic map is a quadratic map; i.e. $f(x)$ is a quadratic ploynomial. 
    It shows a particularly interesting phenomena of 'deterministic chaos' i.e. 
    a deterministic map showing apparently random behaviour.  
    Logistic map has a single parameter, named $\alpha$, and is given by:

    $$f(x)=\alpha x(1-x)$$

    When $0\le \alpha \le 4$, the map takes an input $0\le x \le 1$ to give an output in the same range. 
    For a particular value of the parameter $\alpha$, we want to find the behaviour of the map. 
    For such systems, plot of $x_{n}$ vs $x_{n+1}$ is called phase plot and is a important 
    tool for visualising and analysing such systems. 

    ## Population Dynamics and Spatial heterogeneity

    The logistic map is a mathematical model used to predict population growth and demonstrate chaotic behavior. 

    \paragraph{Guppies example: }
    It applies to a scenario involving an aquarium with guppies, 
    where the question arises: How many guppies can be expected next month if they are left to breed freely?

    The logistic map considers the initial population and its proximity to the 
    maximum capacity of the aquarium. If the population is far from the maximum capacity, 
    indicating sufficient resources, it is likely to increase. 
    Conversely, if the population is close to the maximum capacity, resources will be 
    limited, leading to a decrease in population.

    The logistic map introduces the variable $x$ to represent the proportion of the tank's 
    maximum capacity occupied by guppies. $x$ varies between 0 (no guppies) and 1 (maximum capacity of 100 guppies). i
    
    The formula $rx(1-x)$ estimates the next month's population size, 
    where $r$ is the growth rate determined by factors such as birth and death rates.

    One of the key observations, is that when the value of $r$ increases, above the value of 3,
    the population change settles into a periodic pattern, but beyond $ r = 3.56995$ (approx)

    The population may never settle down into a predictable pattern. 
    But that’s not the worst: two different starting values of $x$ can lead 
    to wildly different predictions of the future, even if they are very, very close.

    ## Logistic regression and Disease mapping on Large moving populations

    Spatial analysis of disease risk, or disease mapping, typically
    relies on information about the residence and health status of
    individuals from population under study. However, residence
    information has its limitations because people are exposed to
    numerous disease risks as they spend time outside of their
    residences. \cite{logistic2} 
    
    Thanks to the wide-spread use of mobile phones and
    GPS-enabled devices, it is becoming possible to obtain a detailed
    record about the movement of human populations. Availability of
    movement information opens up an opportunity to improve the
    accuracy of disease mapping

    The 'alpha' in the logistic equation has its own form in disease mapping,
    formally known as the 'r' factor.

    \paragraph{COVID-19 example: }
    
    When the total number of cases are plotted against a logarithmic scale, 
    we can visualize the exponential growth of COVID-19 cases. 
    Using simple regression analysis (a statistical method to obtain mathematical 
    function using least-square-error method), a model is fitted through the WHO’s 
    data in the log scale. It turns out total number of COVID-19 cases on daily basis can be 
    easily modeled using exponential growth equation (equation 1 given below).     

    $$ n_{i+1} = r * n_i
    
    According to the model fitted to the WHO’s data for cases outside China, 
    the growth rate (r) is ~1.18564. However, since this model is exponential, 
    it diverges to infinity over time, which is physically incorrect. 
    As more people are infected, they are surrounded by people who are already been 
    infected or have developed immunity. So there is a saturation point in 
    the infected population. Typically, in order to predict the growth, another factor is 
    multiplied to the right-hand-side of this exponential growth equation,

    $$ n_{i+1} = r * n_i * ( 1 - n_i/ N) $$

    replace the $n_i/N$ with a variable $x$ and there you go, its just the logistic map equation
    Now Logistic Map is a very simplified, yet very powerful way to model spread of infection like COVID-19. 


# (this is for slides )

    Population Analysis (Leslie Matrix)
        - i Have given you some freedom for this
        - Mostly include what is there in Sir's slides use his resources

    Prey-Predator Model
        - I have attached a CND submission (CS6), take all beautiful looking graphs from it 
        - Take my explanations and make a one liner out of it
        - If you get some 
        - take an example (like an illustration ) wikipedia has a good one (shorten it)
                https://wikiless.org/wiki/Lotka%E2%80%93Volterra_equations?lang=en

    CPG
        - start with hidden Markov Chains - 
            https://yewtu.be/watch?v=RWkHJnFj5rY - abuse this video, take almost all images
                    and mathematical excpressions from this and for explanations just use 
                    transcript of what he says

        -
        https://www.math.clemson.edu/~macaule/classes/f16_math4500/slides/f16_math4500_cpg-islands_handout.pdf
        ( chaap what u like from here include images )
        - for DNA methylation part take a look at the images folder ( include all of those images at least
        once)
        - I will physically explain the coding part and write the description, im
        including the pseudo code here:
        - In the slides, its ok if you rub off theoractical part, i want all
        equations and examples and stuff

    Logistic Maps - same take my CND submission do whatever ( CS6)
        - Do not forget the main part, COVID-19 chaap from this website here's the link
        - https://scribe.rip/analytics-vidhya/modeling-covid-19-spreading-using-data-science-and-logistic-map-8b11cf46c0a8?
        - There is a quote near the end of the above article "TAKE THAT" its very important ( estimation thing)

