import streamlit as st
import pandas as pd
import gensim
from gensim import corpora,models
import numpy as np
import re
import nltk
import gensim

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases, TfidfModel, LsiModel
from gensim.corpora import Dictionary
import gensim.matutils as matutils
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Placeholder lists for countries and UN entities
country_names = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda",
    "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
    "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan",
    "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic",
    "Chad", "Chile", "China", "Colombia", "Comoros", "Congo (Congo-Brazzaville)",
    "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czechia (Czech Republic)", "Democratic Republic of the Congo",
    "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt",
    "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini",
    "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany",
    "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Holy See", "Honduras", "Hungary", "Iceland", "India", "Indonesia",
    "Iran", "Iraq", "Ireland", "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan",
    "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos",
    "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania",
    "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta",
    "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova",
    "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar (formerly Burma)",
    "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger",
    "Nigeria", "North Korea", "North Macedonia (formerly Macedonia)", "Norway", "Oman",
    "Pakistan", "Palau", "Palestine State", "Panama", "Papua New Guinea", "Paraguay",
    "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda",
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa",
    "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles",
    "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia",
    "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan",
    "Suriname", "Sweden", "Switzerland", "Syria", "Tajikistan", "Tanzania", "Thailand",
    "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan",
    "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States of America",
    "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe","un"
]  # Placeholder for country names

un_entities = ["ACABQ-SEC", "ATSMT", "BINUH", "BOA-SEC", "CNMC", "CTED", "DESA",
    "DGACM", "DGC", "DMSPCa", "DOSa", "DPOa", "DPPA", "ECA", "ECE",
    "ECLAC", "EOSG", "ESCAP", "ESCWA", "ETHICS", "GCO", "GOE-DRC", "HSU",
    "IAAC-SEC", "IIIM-Syria", "IM-Myanmar", "IRMCT", "MINUJUSTH", "MINURSO",
    "MINUSCA", "MINUSMA", "MONUSCO", "OAJ", "OCHA", "OCT", "ODA", "OEERC",
    "OHCHR", "OHRLLS", "OICT", "OIOS", "OLA", "OMBUD", "OOSA", "OPESG-WS",
    "OSAA", "OSASG-Cyprus", "OSASG-POG", "OSC SEA", "OSCS", "OSE HoA", "OSEH",
    "OSESG", "OSESG-GL", "OSESG-MYR", "OSESG-SC1559", "OSESG-Syria",
    "OSESG-Yemen", "OSET", "OSRSG-CAAC", "OSRSG-SVC", "OSRSG-VAC", "OVRA",
    "POE-CAR", "POE-DPRK", "POE-Libya", "POE-Mali", "POE-S. Sudan", "POE-SOM",
    "POE-Sudan", "POE-Yemen", "RCNYO", "RCS", "RSCE", "SCR 2231", "UN75",
    "UNAKRT", "UNAMA", "UNAMI", "UNAMID", "UNCC", "UNCTAD", "UNDOF", "UNDRR",
    "UNEP", "UNFICYP", "UN-Habitat", "UNIFIL", "UNIOGBIS", "UNISFA", "UNITAD",
    "UNITAMS", "UNLB", "UNMHA", "UNMIK", "UNMISS", "UNMOGIP", "UNOAU", "UNOCA",
    "UNODC", "UNOG", "UNOMS", "UNON", "UNOP", "UNOV", "UNOWAS", "UNRCCA",
    "UNRGID", "UNROD", "UNSCO", "UNSCOL", "UNSMIL", "UNSOM", "UNSOS",
    "UN-TBLDC", "UNTSO", "UNVMC",'etc',"analyze","prepare","participate","undertake","initiate","undertakes","generates","client",'identifies','mission','clients','initiate','initiates','prepares']  # Placeholder for UN entities and terms

# Function to remove country names
def remove_country_names(text):
    pattern = r'\b(?:' + '|'.join(re.escape(name) for name in country_names) + r')\b'
    return re.sub(pattern, '', text, flags=re.IGNORECASE)

# Function to replace entity names with "the organization"
def remove_entity_names(text):
    pattern = r'\b(?:' + '|'.join(re.escape(name) for name in un_entities) + r')\b'
    return re.sub(pattern, 'the organization', text, flags=re.IGNORECASE)

# Streamlit app setup
st.title('Upload CSV and Preprocess Text')

st.write('Please upload your CSV file.')

uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'RESPONSIBILITIES' in df.columns:
        # Apply removal of country names and entity replacement
        df['RESPONSIBILITIES'] = df['RESPONSIBILITIES'].apply(remove_country_names)
        df['RESPONSIBILITIES'] = df['RESPONSIBILITIES'].apply(remove_entity_names)

        # Convert text into list format for further processing
        docs = df['RESPONSIBILITIES'].tolist()
        
        # Tokenize the documents
        tokenizer = RegexpTokenizer(r'\w+')
        docs = [tokenizer.tokenize(doc.lower()) for doc in docs]
        
        # Remove numbers and stopwords
        stop_words = set(stopwords.words('english'))
        docs = [[token for token in doc if not token.isnumeric() and token not in stop_words] for doc in docs]
        
        # Remove words that are only one character
        docs = [[token for token in doc if len(token) > 1] for doc in docs]
        
        # Lemmatize the documents
        lemmatizer = WordNetLemmatizer()
        docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
        
        # Compute bigrams
        bigram = Phrases(docs, min_count=10)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    docs[idx].append(token)
        
        # Remove rare and common tokens
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=10, no_above=0.8)
        docs = [[token for token in doc if token in dictionary.token2id] for doc in docs]
        
        # Bag-of-words representation of the documents
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        word_counts = [[(dictionary[id], count) for id, count in line] for line in corpus]
        
        # Generate a sorted list of unique tokens
        sorted_tokens = sorted(dictionary.items(), key=lambda k: k[0], reverse=False)
        unique_tokens = [token for (ID, token) in sorted_tokens]
        
        matrix = gensim.matutils.corpus2dense(corpus,num_terms=len(dictionary),dtype = 'int')

        matrix = matrix.T #transpose the matrix 
    

        #convert the numpy matrix into pandas data frame
        matrix_df = pd.DataFrame(matrix, columns=unique_tokens)
       
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        # Tfidf Transformation 
        #from gensim.models import LsiModel
        tfidf = models.TfidfModel(corpus) #fit tfidf model
        corpus_tfidf = tfidf[corpus]      #apply model to the corpus 
        
        lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=6)
        corpus_lsi = lsi[corpus_tfidf]

        # Convert corpus to dense matrix for clustering
        lsi_corpus_dense = np.array([matutils.sparse2full(doc, lsi.num_topics) for doc in corpus_lsi])  
        # Function to calculate silhouette scores and find optimal clusters
        def find_optimal_clusters(data, start=2, end=10, threshold=0.2):
            silhouette_scores = {}
            for n_clusters in range(start, end + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(data)
                score = silhouette_score(data, labels)
                silhouette_scores[n_clusters] = score
            
            optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
            return silhouette_scores, optimal_clusters

        silhouette_scores, optimal_clusters = find_optimal_clusters(lsi_corpus_dense)


        
        
        # Display results
        st.write("### Silhouette Scores by Number of Clusters")
        for n_clusters, score in silhouette_scores.items():
            st.write(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}")

        st.write("### Optimal Number of Clusters")
        st.write(f"Optimal clusters: {optimal_clusters}, with a silhouette score of {silhouette_scores[optimal_clusters]:.4f}")
        
        from gensim.models import LsiModel

        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary.id2token, num_topics=optimal_clusters)
        topics_words = []
        for topic_num, topic in enumerate(lsi.show_topics(num_topics=optimal_clusters, num_words=20)):
        
            top_words = {word.strip('"') for word in re.findall('"([^"]*)"', topic[1])}

            topics_words.append(top_words)
        
        if 'optimal_clusters' in locals() or 'optimal_clusters' in globals():
            lsi = LsiModel(corpus_tfidf, id2word=dictionary.id2token, num_topics=optimal_clusters)
            
            # Extract top words for each topic
            topics_words = []  # Stores the top words for each topic
            for i in range(optimal_clusters):
                # Extract the top words for each topic (cluster)
                top_words = set(word for word, _ in lsi.show_topic(i, topn=10))
                topics_words.append(top_words)
            
            # Function to calculate the percentage overlap between two sets of words
            def calculate_percentage_overlap(top_words1, top_words2):
                intersection = len(top_words1.intersection(top_words2))
                total_words = len(top_words1)  # Calculating how many from top_words1 are in top_words2
                return (intersection / total_words) * 100 if total_words > 0 else 0
            
            # Display the overlap percentage for each cluster pair in Streamlit
            st.write("## Overlap Percentages Between Clusters")
            overlap_data = []  # To store overlap data for display
            for i in range(len(topics_words)):
                for j in range(i + 1, len(topics_words)):  # Avoid comparing a cluster with itself or repeating comparisons
                    overlap_percentage = calculate_percentage_overlap(topics_words[i], topics_words[j])
                    overlap_data.append({"Cluster 1": i + 1, "Cluster 2": j + 1, "Overlap (%)": f"{overlap_percentage:.2f}%"})

            # Convert overlap data to a pandas DataFrame for nicer display
            overlap_df = pd.DataFrame(overlap_data)
            st.table(overlap_df)
        
        

        # Ask the user for a threshold percentage for merging clusters
        threshold_percentage = st.number_input("Enter the percentage threshold for merging clusters:",
                                                min_value=0, max_value=100)
        st.write("You set the threshold to:", threshold_percentage, "%")

        def merge_clusters_on_overlap(topics_words, threshold=threshold_percentage):
            cluster_index = 0
            merged_clusters = []
            merge_info = []
            original_cluster_count = len(topics_words)
            while cluster_index < original_cluster_count:
                current_cluster = topics_words[cluster_index]
                clusters_to_merge = [cluster_index]
                for i, other_cluster in enumerate(topics_words):
                    if i != cluster_index and i not in clusters_to_merge:
                        overlap_percentage = calculate_percentage_overlap(current_cluster, other_cluster)
                        if overlap_percentage > threshold_percentage:
                            current_cluster = current_cluster.union(other_cluster)
                            clusters_to_merge.append(i)
                
                # Update the list of topics words by removing merged clusters
                topics_words = [cluster for i, cluster in enumerate(topics_words) if i not in clusters_to_merge[1:]]
                original_cluster_count = len(topics_words)  # Update the count as clusters are merged
                
                merged_clusters.append(current_cluster)
                merge_info.append(clusters_to_merge)
                cluster_index += 1  # Move to the next cluster

            return merged_clusters, merge_info

        # Execute the merge process using the user-defined threshold
        merged_clusters, merge_info = merge_clusters_on_overlap(topics_words, threshold=threshold_percentage)

        # Displaying merged clusters and their information in Streamlit
        st.write("## Merged Clusters and Merge Information")
        for i, (cluster, indices) in enumerate(zip(merged_clusters, merge_info)):
            cluster_words = ', '.join(sorted(cluster))  # Sort and join the cluster words for display
            original_clusters = ', '.join(map(str, [index + 1 for index in indices]))  # Map indices to cluster numbers
            st.write(f"Merged Cluster #{i + 1}: {cluster_words}")
            st.write(f" - Formed from original clusters: {original_clusters}")

        

        # Create mapping from original topic indices to merged cluster indices
        topic_to_merged_cluster = {}
        for merged_index, original_indices in enumerate(merge_info):
            for original_index in original_indices:
                topic_to_merged_cluster[original_index] = merged_index
        top_words_per_topic = {f'Merged Cluster {i+1}': list(cluster) for i, cluster in enumerate(merged_clusters)}

        # Display the words in each merged cluster
        for cluster_key, words_list in top_words_per_topic.items():
            print(f"{cluster_key}: {', '.join(words_list)}")
                
        # Map each document to its dominant topic
        doc2topic = [max(lsi[doc], key=lambda item: abs(item[1]))[0] for doc in corpus_tfidf]

        # Update the DataFrame to reflect merged clusters instead of original dominant topics
        df['Dominant_Topic'] = doc2topic
        df['Merged_Cluster'] = df['Dominant_Topic'].apply(lambda x: topic_to_merged_cluster.get(x))

        # Ensure 'JO_Number' column exists in df
        if 'JO_Number' not in df.columns:
            st.error("DataFrame does not contain 'JO_Number'. Cannot group by JO Numbers.")
        else:
            # Group JO numbers by merged cluster
            merged_cluster_to_jo = df.groupby('Merged_Cluster')['JO_Number'].apply(lambda x: list(set(x))).to_dict()

            # Display JO numbers for each merged cluster in Streamlit
            st.write("## JO Numbers for Merged Clusters")
            for merged_cluster_id, jo_list in sorted(merged_cluster_to_jo.items()):
                st.markdown(f"**Merged Cluster #{merged_cluster_id + 1}:** {', '.join(map(str, jo_list))}")
        
        
        def get_job_title_by_jo_number(jo_number, dataframe):
            """
            Function to return the job title given a JO_Number.
            
            :param jo_number: The JO_Number to search for.
            :param dataframe: The dataframe containing the job information.
            :return: The job title corresponding to the given JO_Number, or a not found message.
            """
            # Find the row where the JO_Number matches
            matching_row = dataframe[dataframe['JO_Number'] == jo_number]
            
            # Check if there's a match
            if not matching_row.empty:
                # Return the job title
                return matching_row['RESPONSIBILITIES'].iloc[0]
            else:
                # Return a message if the JO_Number is not found
                return "Job title not found for the given JO_Number."

        # Ask the user for a JO_Number as input
        jo_number_example = st.text_input("Enter a JO_Number to search for its job responsibilities:")

        # Check if the user has entered a value before proceeding
        if jo_number_example:
        
            jo_number_int = int(jo_number_example)
            job_title = get_job_title_by_jo_number(jo_number_int , df)
            # Display the job title or not found message
            st.write(f"Job Responsibilities: {job_title}")
        else:
            st.write("Please enter a JO_Number above.")
    
        
        template1 = """ 
        You are an expert in human resources analyzing job descriptions for the role of Programme Management Officer.
        These are examples of the type of specialties you can assign to this role:
        * Disaster Risk Management
        * Data Visualization Expert
        * Strategic Implementation Advisor
        * Climate Adaptation Coordinator
        Based on these top terms listed found in the responsibilities of the job description: {cluster}
        * Generate only one Specialty label, maximum 4 words.

        """
        # Initialize the LLM with your chosen model and settings
        from langchain.llms import Ollama
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        llm = Ollama(model="mistral-openorca", temperature=0.9)

        # Define the prompt template with the specified instructions
        # Replace 'Your prompt here with cluster variable: {cluster}' with your actual template
        prompt_template = PromptTemplate(
            input_variables=["cluster"],
            template=template1
        )

        # Initialize the LLM Chain with the prompt template
        chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)

        # Assuming top_words_per_topic is defined and contains your clusters' top words
        # Iterate over each merged cluster in top_words_per_topic to generate content
        st.write("## Generated Specialty Labels for Merged Clusters")
        for cluster_key, words_list in top_words_per_topic.items():
            # Prepare the cluster words as a string
            cluster_str = ", ".join(words_list)
            
            # Generate the specialty label using the LLM chain
            generated_label = chain.run({"cluster": cluster_str})

            # Display the generated specialty label for the cluster in Streamlit
            st.markdown(f"**{cluster_key}:** {generated_label}")
            st.text("-" * 50)

    else:
        st.error("The uploaded CSV does not contain the 'RESPONSIBILITIES' column.")