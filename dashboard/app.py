# import json
# import matplotlib.pyplot as plt
# import pandas as pd
# import streamlit as st
# import seaborn as sns
# import numpy as np
# import matplotlib.image as mpimg
# import datetime as dt
# from io import BytesIO
# import base64
# from config import CONFIG
# import plotly.express as px
# import plotly.graph_objects as go
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# from PIL import Image
# # New imports
# from bertopic import BERTopic
# from google.cloud import aiplatform



# # Load acutal articles scrapped JSON data from file
# # with open(CONFIG["data_file"], 'r') as file:
# #     data = json.load(file)
# df_articles = pd.read_csv(
#     CONFIG["articles_csv_file"],
#     parse_dates=["date"],       # auto‐parse your date column
#     dtype={"source": str,       # ensure string type
#            "domain": str,
#            "content": str}
# )

# # drop any rows with missing key fields
# df_articles = df_articles.dropna(subset=["date", "source", "content"])

# # # Load Summary JSON data from file
# # with open(CONFIG["summaries_file"], 'r') as file:
# #     summaries_data = json.load(file)


# # Load country of origin wrt news website JSON data from file
# with open(CONFIG["country_origin_file"], 'r') as file:
#     country_origin_data = json.load(file, strict=False)


# ## BERTopic 
# # 1) Prepare the list of documents
# docs = df_articles["content"].tolist()

# # 2) Fit BERTopic
# topic_model = BERTopic(language="english", nr_topics="auto")
# topics, probs = topic_model.fit_transform(docs)

# # 3) Pull out a “representative snippet” for each topic
# #    We’ll join the top 10 words for each topic into a prompt
# topic_info = topic_model.get_topic_info()            # has topic IDs & counts
# topic_repr = {
#     t_id: " ".join([word for word, _ in topic_model.get_topic(t_id)])
#     for t_id in topic_info.Topic.unique() if t_id != -1
# }


# # 1a) Group the raw article texts by topic
# topic_to_texts = {}
# for topic_id, text in zip(topics, df_articles["content"]):
#     # skip outliers
#     if topic_id == -1:
#         continue
#     topic_to_texts.setdefault(topic_id, []).append(text)

# import re

# def extract_sentences_for_topic(texts: list[str], keywords: list[str]) -> str:
#     """
#     From a list of documents, pull out every sentence that contains
#     at least one of the keywords. Returns them concatenated.
#     """
#     # build a case-insensitive regex that looks for any of the words
#     # e.g. r"\b(word1|word2|word3)\b"
#     pattern = re.compile(r"\b(" + "|".join(map(re.escape, keywords)) + r")\b", 
#                          flags=re.IGNORECASE)

#     selected = []
#     for doc in texts:
#         # naive sentence split
#         for sent in re.split(r'(?<=[\.\?\!])\s+', doc):
#             if pattern.search(sent):
#                 selected.append(sent.strip())
#     # join them with spaces (or newlines) up to a reasonable length
#     return " ".join(selected)

# aiplatform.init(
#     project=CONFIG["vertex_project"],
#     location=CONFIG["vertex_region"],
# )
# summarizer = aiplatform.Endpoint(
#     CONFIG["summarizer_endpoint"]
# )

# # Build a map topic_id → list of URLs
# topic_to_urls = {}
# for topic_id, url in zip(topics, df_articles["url"]):
#     topic_to_urls.setdefault(topic_id, []).append(url)

# summaries_data = []
# for topic_id, repr_text in topic_repr.items():
#     # call Vertex AI summarizer
#     resp = summarizer.predict(
#         instances=[{"content": repr_text}],
#         parameters={"temperature": 0.0, "maxOutputTokens": 256},
#     )
#     summary = resp.predictions[0].get("summary", "")
    
#     # grab up to 5 example URLs
#     urls = topic_to_urls.get(topic_id, [])[:5]

#     summaries_data.append({
#         "topic_id": topic_id,
#         "summary": summary,
#         "words": repr_text.split()[:10],
#         "urls": urls,
#     })

# # Setting layout for dashboard
# st.set_page_config(page_title = 'ELMO Dashboard',
#                     page_icon="🗞️",
#                     layout="wide",
#                 )  

# st.markdown(
#     """
#     <style>
#     .main .block-container {
#         padding-top: 0.5rem;
#         padding-bottom: 0.5rem;
#         padding-left: 1rem;
#         padding-right: 1rem;
#     }
#     .stButton button {
#         background-color: #4CAF50;
#         color: white;
#         border: none;
#         padding: 5px 10px;
#         text-align: center;
#         text-decoration: none;
#         display: inline-block;
#         font-size: 12px;
#         margin: 2px 1px;
#         cursor: pointer;
#         border-radius: 4px;
#     }
#     .small-row {
#         padding-top: 0.1rem;
#         padding-bottom: 0.1rem;
#     }
#     .large-row {
#         padding-top: 2rem;
#         padding-bottom: 2rem;
#     }
#     h1, h2, h3, h4, h5, h6 {
#         font-size: 1rem;
#         margin-bottom: 0.5rem;  # Reduce margin-bottom for headers
#     }
#     .stMarkdown p {
#         font-size: 0.8rem;
#     }
#     [data-testid="stVerticalBlock"] > div {
#         margin-bottom: 0rem; /* Reduce space between rows */
#     }
#     .keyword-container {
#     display: flex;
#     justify-content: center;
#     align-items: center;
#     height: 100%;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# st.image('ELMO_dashboard_header.PNG', use_container_width = True)

# # Layout - Two Columns
# col1, col2, col3 = st.columns([2,1,1])
# col4, col5 = st.columns([2, 2])


# # 1. Data Extraction
# articles = data['articles']

# df_articles = pd.DataFrame(articles)

# df_articles = df_articles.dropna()

# print(df_articles['domain'].value_counts())

# # article_sources = [article.split('/')[0] for article in df_articles['source']]

# # article_dates = [dt.datetime.strptime(article, '%Y-%m-%d %H:%M:%S') for article in df_articles['date']]

# article_dates = df_articles["date"]
# article_sources = df_articles["source"].str.split("/", n=1).str[0]


# # articles_total = data['articles']
# # article_count_total = len(articles_total)
# # article_sources_total = [article['source'].split('/')[0] for article in articles_total]
# # article_dates_total = [dt.datetime.strptime(article['date'], '%Y-%m-%d %H:%M:%S') for article in articles_total]
# # article_contents_total = [article['content'] for article in articles_total]


# # # 2. Extracting Query keywords from phrase_scores
# # phrase_scores = data["crawler"]["phrase_scores"]

# # for phrase, score in phrase_scores.items():
# #     print(f"{phrase}: {score}")

# # keywords = list(phrase_scores.keys())

# keywords = CONFIG["keywords"]

# # 3. Create a dictionary to map URLs to source names
# url_to_name = {entry['website']: entry['name'] for entry in country_origin_data}


# df = pd.DataFrame({
#     'Date': article_dates,
#     'Source': article_sources
# })

# # 4. Counting articles per month
# df['Month'] = df['Date'].dt.to_period('M')
# article_counts_per_month = df['Month'].value_counts().sort_index()




# #### Functions
# # Function to get country from website
# def get_country_from_website(website):
#     for entry in country_origin_data:
#         if entry['website'] == website:
#             return entry['country']
#     print(website, 'other')
#     return "Other"

# # Function to Add Images to Bars
# def add_flag(img_path, x, y, ax):
#     img = mpimg.imread(img_path)
#     imagebox = OffsetImage(img, zoom=0.6)  # Adjust zoom level for better fit
#     ab = AnnotationBbox(imagebox, (x, y), frameon=False, xycoords="data", boxcoords="offset points", pad=0)
#     ax.add_artist(ab)

# # Function to get flag image path
# def get_flag_image_path(country):
#     return f"flags/{country}_flag.png"

# # Function to read and encode image to base64
# def get_base64_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode()
    
# # Function to get source name from website
# def get_source_name(website):
#     return url_to_name.get(website, website)

# # Function to remove 'localhost' part from the URL
# def clean_url(full_url):
#     if not full_url.startswith('http://') and not full_url.startswith('https://'):
#         full_url = 'http://' + full_url
#     return full_url



# #### 

# ## PLOT FOR ARTICLES SOURCED
# st.markdown('<div class="small-row">', unsafe_allow_html=True)
# with col1:

#     # Function to get source name from website
#     def get_source_name(website):
#         return url_to_name.get(website, website)

#     # Bar Plot Data
#     source_counts = pd.DataFrame({'Source': article_sources})
#     source_counts['Country'] = source_counts['Source'].apply(get_country_from_website)
#     source_counts['SourceName'] = source_counts['Source'].apply(get_source_name)
#     source_counts = source_counts.groupby(['SourceName', 'Country']).size().reset_index(name='Count')

#     # Bar Plot with Flags and Source Names in col1
#     st.header("Articles Sourced" )
#     fig = go.Figure()

#     bar_data = sorted(zip(source_counts['SourceName'], source_counts['Count'], source_counts['Country']), key= lambda x: x[1])

#     print(bar_data)

#     for source_name, count, country in bar_data:
#         fig.add_trace(go.Bar(
#             y=[source_name],
#             x=[count],
#             orientation='h',
#             marker=dict(color='lightblue'),
#             width=0.9 
#         ))

#     fig.update_layout(
#         xaxis_title='',
#         yaxis_title='',
#         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         showlegend=False,
#         plot_bgcolor='white',
#         bargap=0,
#         margin=dict(t=0, b=0, l=0, r=0),
#         height=220,  # Adjust height as needed
#         width=800    # Adjust width as needed
#     )

#     # Add flag images as annotations
#     for source_name, count, country in zip(source_counts['SourceName'], source_counts['Count'], source_counts['Country']):
#         image_path = get_flag_image_path(country)
#         img_str = get_base64_image(image_path)
#         fig.add_layout_image(
#             dict(
#                 source=f"data:image/png;base64,{img_str}",
#                 x=count-1,
#                 y=source_name,
#                 xref="x",
#                 yref="y",
#                 xanchor="right",
#                 yanchor="middle",
#                 sizex=15,  # Adjust size as needed
#                 sizey=1.5,
#                 sizing="contain",
#                 opacity=1,
#                 layer="above"
#             )
#         )
#         # Add total number of articles at the end of each bar
#         fig.add_annotation(
#             x=count+1,
#             y=source_name,
#             text=f"{count}",
#             showarrow=False,
#             xanchor="left",
#             yanchor="middle",
#             font=dict(size=16, color="black")
#         )

#         # Add y-axis labels inside the bars
#         fig.add_annotation(
#             x=2,  # Position slightly inside the bar
#             y=source_name,
#             text=source_name,
#             showarrow=False,
#             xanchor="left",
#             yanchor="middle",
#             font=dict(size=16, color="black")
#         )

#     st.plotly_chart(fig)
#     st.markdown('</div>', unsafe_allow_html=True)


# ## KEYWORD SECTION
# with col2:
#     keyword = keywords[0] 
#     st.header("Keywords")
#     st.markdown(f"""
#     <div class="small-row keyword-container" style='background-color: lightblue; padding: 5px; font-size: 50px; font-family: Helvetica; color: white; border-radius: 15px;'>
#     {keyword}
#     </div>
#     """, unsafe_allow_html=True)


# ## CUSTOM IMAGE FOR KEYWORD
# with col3:
#     st.text("")
    
#     # File uploader for image upload
#     uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="uploader")

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         buffered = BytesIO()
#         image.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode()
#         st.markdown(f"""
#         <div class="small-row" style="padding: 20px; border-radius: 15px; background-color: lightgrey; display: flex; justify-content: center;">
#             <img src="data:image/png;base64,{img_str}" style="max-width: 80%; border-radius: 15px;">
#         </div>
#         """, unsafe_allow_html=True)
   
#         st.markdown("""
#         <style>
#         .stFileUploader {display: none;}
#         </style>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown("""
#         <div class="small-row" style="padding: 20px; border-radius: 15px; background-color: lightgrey; display: flex; justify-content: center; align-items: center;">
#             <p style="font-size: 20px; color: grey;">Drag and drop an image here or click to upload</p>
#         </div>
#         """, unsafe_allow_html=True)
#         \


# ## NUMBER OF ARTICLES PER TIME FRAME PLOT
# with col4:
#     # Dynamic time scale selection
#     time_scale = CONFIG['TIME_SCALE']

#     today = dt.datetime.now()

#     st.header(f"Number of Articles per {time_scale}")

#     if time_scale == "Day":
#         start_date = today - dt.timedelta(days=60)
#         mask = (df['Date'] <= today) & (df['Date'] >= start_date)
#         df = df.loc[mask]
#         df['TimeScale'] = df['Date'].dt.to_period('D')
#     elif time_scale == "Month":
#         start_date = today - dt.timedelta(weeks=52)
#         mask = (df['Date'] <= today) & (df['Date'] >= start_date)
#         df = df.loc[mask]
#         df['TimeScale'] = df['Date'].dt.to_period('M')
#     elif time_scale == "Year":
#         df['TimeScale'] = df['Date'].dt.to_period('Y')

#     # print(df['TimeScale'])

#     article_counts_per_timescale = df['TimeScale'].value_counts().sort_index()

#     # print(article_counts_per_timescale)

#     fig = go.Figure()

#     fig.add_trace(go.Bar(
#         x=article_counts_per_timescale.index.astype(str),
#         y=article_counts_per_timescale.values,
#         marker=dict(color='lightblue')
#     ))

#     index = article_counts_per_timescale.index
#     tick_vals = index.astype(str)

#     if time_scale == "Day":
#         tick_text = [t.strftime('%b %d') for t in index]
#     elif time_scale == "Month":
#         tick_text = [t.strftime('%b %y') for t in index]
#     elif time_scale == "Year":
#         tick_text = [t.strftime('%Y') for t in index]

#     # print(tick_vals)
#     # print(tick_text)

#     fig.update_layout(
#         title=f'',
#         xaxis=dict( tickmode='array',
#                     tickvals=tick_vals,
#                     ticktext=tick_text,
#                     tickangle=45,
#                     tickfont=dict(size=15),
#                     nticks=len(article_counts_per_timescale) 
#         ),
#         yaxis=dict(tickfont=dict(size=15)),
#         plot_bgcolor='white',
#         margin=dict(t=40, b=40, l=40, r=40)
#     )
#     st.plotly_chart(fig)


# ## SUMMARY SECTION
# with col5:
#     st.header("Summaries")
#     for i, item in enumerate(summaries_data[:5]):
#         st.markdown(
#             f"<div class='summary-container'><p style='background-color: #f2f2f2; padding: 5px; border-radius: 15px; font-family: Helvetica; font-size: 0.8rem;'>{item['summary']}</p></div>",
#             unsafe_allow_html=True
#         )
        
#         source_links = " ".join(
#             [f"<a href='{clean_url(source['url'])}' style='padding: 5px; font-family: Helvetica; margin-right: 5px; border-radius: 10px; background-color: lightblue; display: inline-block;' target='_blank'>{url_to_name.get(source['url'].split('/')[0], 'Source')}</a>" 
#              for j, source in enumerate(item['sources'])]
#         )
#         st.markdown(source_links, unsafe_allow_html=True)

print("HELLO")

import re
import json
import datetime as dt
from io import BytesIO
import base64

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from bertopic import BERTopic
from google.cloud import aiplatform

from config import CONFIG
import logging

# at top of app.py
import logging

try:

    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()]  # force stdout
    )
    logger = logging.getLogger(__name__)
    logger.info("🔄 Starting topic modelling…")


    # 5a) Fit BERTopic
    print(f"• Fitting BERTopic on {len(docs)} documents…")             # ←
    topic_model = BERTopic(language="english", nr_topics="auto")
    topics, probs = topic_model.fit_transform(docs)
    print("✅ BERTopic fit complete. Found topics:", set(topics))      # ←

    # get top words per topic
    topic_info = topic_model.get_topic_info()
    topic_repr = {
        tid: " ".join([w for w,_ in topic_model.get_topic(tid)])
        for tid in topic_info.Topic.unique() if tid != -1
    }

    # 5b) Build maps topic→texts & topic→urls
    print("• Grouping texts and URLs by topic…")                       # ←
    topic_to_texts = {}
    topic_to_urls  = {}
    for tid, text, url in zip(topics, df_articles["content"], df_articles["url"]):
        if tid == -1: continue
        topic_to_texts.setdefault(tid, []).append(text)
        topic_to_urls .setdefault(tid, []).append(url)
    print("✅ Grouped into", len(topic_to_texts), "topics.")           # ←

    # 5c) Init Vertex AI summarizer
    print("• Initializing Vertex AI…")                                # ←
    aiplatform.init(project=CONFIG["vertex_project"], location=CONFIG["vertex_region"])
    summarizer = aiplatform.Endpoint(CONFIG["summarizer_endpoint"])
    print("✅ Vertex AI endpoint ready.")                              # ←

    # ——————————————————————————————
    #  Helper functions
    # ——————————————————————————————
    def get_country_from_website(website: str) -> str:
        for entry in country_origin_data:
            if entry['website'] == website:
                return entry['country']
        return "Other"

    def get_source_name(website: str) -> str:
        return url_to_name.get(website, website)

    def clean_url(full_url: str) -> str:
        if not full_url.startswith(('http://', 'https://')):
            full_url = 'http://' + full_url
        return full_url

    def extract_sentences_for_topic(texts: list[str], keywords: list[str]) -> str:
        """Pull out every sentence that contains at least one keyword."""
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, keywords)) + r")\b", flags=re.IGNORECASE)
        selected = []
        for doc in texts:
            for sent in re.split(r'(?<=[\.\?\!])\s+', doc):
                if pattern.search(sent):
                    selected.append(sent.strip())
        return " ".join(selected)


    # ——————————————————————————————
    #  Load auxiliary data
    # ——————————————————————————————
    with open(CONFIG["country_origin_file"], 'r', encoding='utf-8') as f:
        country_origin_data = json.load(f, strict=False)

    url_to_name = {e['website']: e['name'] for e in country_origin_data}


    # ——————————————————————————————
    #  Load articles from CSV
    # ——————————————————————————————
    df_articles = pd.read_csv(
        CONFIG["articles_csv_file"],
        parse_dates=["date"],
        dtype={"source": str, "url": str, "content": str}
    ).dropna(subset=["date", "source", "content"])


    # ——————————————————————————————
    #  Page config & CSS
    # ——————————————————————————————
    st.set_page_config(page_title='ELMO Dashboard', page_icon="🗞️", layout="wide")
    st.markdown("""
        <style>
        .main .block-container { padding: 1rem; }
        h1,h2,h3,h4 { margin-bottom:0.25rem; }
        .stMarkdown p { font-size:0.85rem; }
        .keyword-container { display:flex; justify-content:center; align-items:center; height:100%; }
        </style>
    """, unsafe_allow_html=True)

    st.image('ELMO_dashboard_header.PNG', use_container_width=True)


    # ——————————————————————————————
    #  Layout: columns
    # ——————————————————————————————
    col1, col2, col3 = st.columns([2,1,1])
    col4, col5       = st.columns([2,2])


    # =============================================================================
    # 1) ARTICLES SOURCED BAR (col1)
    # =============================================================================
    # derive “source” name & country
    df_articles["article_source"] = df_articles["source"].str.split("/", n=1).str[0]
    tmp = df_articles[["article_source"]].copy()
    tmp["Country"]     = tmp["article_source"].apply(get_country_from_website)
    tmp["SourceName"]  = tmp["article_source"].apply(get_source_name)
    source_counts     = tmp.groupby(["SourceName","Country"]).size().reset_index(name="Count")

    with col1:
        st.header("Articles Sourced")
        fig = go.Figure()
        bar_data = sorted(
            zip(source_counts["SourceName"], source_counts["Count"], source_counts["Country"]),
            key=lambda x: x[1]
        )
        for name, cnt, country in bar_data:
            fig.add_trace(go.Bar(
                y=[name], x=[cnt], orientation='h',
                marker=dict(color='lightblue'), width=0.9
            ))
        # add flags & annotations
        for name, cnt, country in zip(source_counts["SourceName"], source_counts["Count"], source_counts["Country"]):
            # base64 encode flag
            with open(f"flags/{country}_flag.png","rb") as imgf:
                img_b64 = base64.b64encode(imgf.read()).decode()
            fig.add_layout_image(dict(
                source=f"data:image/png;base64,{img_b64}",
                x=cnt-1, y=name, xref="x", yref="y",
                xanchor="right", yanchor="middle", sizex=15, sizey=1.5, sizing="contain"
            ))
            # total count label
            fig.add_annotation(x=cnt+1, y=name, text=str(cnt),
                            showarrow=False, xanchor="left", yanchor="middle")
            # name inside bar
            fig.add_annotation(x=2, y=name, text=name,
                            showarrow=False, xanchor="left", yanchor="middle",
                            font=dict(color="black"))
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white', margin=dict(t=0,b=0,l=0,r=0), height=220, width=800
        )
        st.plotly_chart(fig, use_container_width=True)


    # =============================================================================
    # 2) KEYWORD (col2)
    # =============================================================================
    with col2:
        st.header("Keywords")
        kw = CONFIG["keywords"][0]
        st.markdown(f"""
        <div class="keyword-container" style="background-color:lightblue;
            padding:10px; border-radius:10px; font-size:2rem; color:white;">
            {kw}
        </div>
        """, unsafe_allow_html=True)


    # =============================================================================
    # 3) IMAGE UPLOADER (col3)
    # =============================================================================
    with col3:
        st.text("")
        upl = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
        if upl:
            img = Image.open(upl)
            buf = BytesIO(); img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"""
            <div style="padding:1rem; background:#eee; border-radius:10px;
                        display:flex; justify-content:center;">
                <img src="data:image/png;base64,{img_b64}" style="max-width:100%; border-radius:10px;" />
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding:2rem; background:#eee; border-radius:10px;
                        text-align:center; color:#777;">
                Drag & drop an image here<br/>or click to upload
            </div>
            """, unsafe_allow_html=True)


    # =============================================================================
    # 4) ARTICLES OVER TIME (col4)
    # =============================================================================
    with col4:
        st.header(f"Number of Articles per {CONFIG['TIME_SCALE']}")
        today = dt.datetime.now()
        df = df_articles.copy()
        if CONFIG['TIME_SCALE']=="Day":
            df = df[df['date']>=today-dt.timedelta(days=60)]
            df['TS'] = df['date'].dt.to_period('D')
        elif CONFIG['TIME_SCALE']=="Month":
            df = df[df['date']>=today-dt.timedelta(weeks=52)]
            df['TS'] = df['date'].dt.to_period('M')
        else:
            df['TS'] = df['date'].dt.to_period('Y')

        counts = df['TS'].value_counts().sort_index()
        fig2 = go.Figure([go.Bar(
            x=counts.index.astype(str),
            y=counts.values,
            marker=dict(color='lightblue')
        )])
        fig2.update_layout(
            xaxis=dict(tickangle=45, tickfont=dict(size=12)),
            plot_bgcolor='white', margin=dict(t=20,b=20,l=20,r=20)
        )
        st.plotly_chart(fig2, use_container_width=True)


    # =============================================================================
    # 5) TOPIC MODELING & SUMMARIZATION (col5)
    # =============================================================================
    print("🔄 5) Starting topic‐modeling & summarization…")            # ←

    # 5a) Fit BERTopic
    print(f"• Fitting BERTopic on {len(docs)} documents…")             # ←
    topic_model = BERTopic(language="english", nr_topics="auto")
    topics, probs = topic_model.fit_transform(docs)
    print("✅ BERTopic fit complete. Found topics:", set(topics))      # ←

    # get top words per topic
    topic_info = topic_model.get_topic_info()
    topic_repr = {
        tid: " ".join([w for w,_ in topic_model.get_topic(tid)])
        for tid in topic_info.Topic.unique() if tid != -1
    }

    # 5b) Build maps topic→texts & topic→urls
    print("• Grouping texts and URLs by topic…")                       # ←
    topic_to_texts = {}
    topic_to_urls  = {}
    for tid, text, url in zip(topics, df_articles["content"], df_articles["url"]):
        if tid == -1: continue
        topic_to_texts.setdefault(tid, []).append(text)
        topic_to_urls .setdefault(tid, []).append(url)
    print("✅ Grouped into", len(topic_to_texts), "topics.")           # ←

    # 5c) Init Vertex AI summarizer
    print("• Initializing Vertex AI…")                                # ←
    aiplatform.init(project=CONFIG["vertex_project"], location=CONFIG["vertex_region"])
    summarizer = aiplatform.Endpoint(CONFIG["summarizer_endpoint"])
    print("✅ Vertex AI endpoint ready.")                              # ←

    # 5d) Summarize each topic
    summaries_data = []
    for tid, repr_words in topic_repr.items():
        print(f"→ Topic {tid}: extracting sentences…")                # ←
        keywords = repr_words.split()[:10]
        txt = extract_sentences_for_topic(topic_to_texts.get(tid, []), keywords)
        if not txt:
            txt = repr_words
            print(f"   (no matching sentences; using word‐bag fallback)")

        print(f"→ Topic {tid}: calling summarizer…")                 # ←
        resp = summarizer.predict(
            instances=[{"content": txt}],
            parameters={"temperature":0.0, "maxOutputTokens":256},
        )
        summ = resp.predictions[0].get("summary", "")
        print(f"✅ Topic {tid} summary received: {summ[:50]}…")       # ←

        urls = topic_to_urls.get(tid, [])[:5]
        summaries_data.append({"topic_id":tid, "summary":summ, "urls":urls})

    print("🎉 Summarization complete for all topics.")                # ←

    # 5e) Render summaries
    with col5:
        st.header("Topic Summaries")
        for item in summaries_data:
            st.subheader(f"Topic #{item['topic_id']}")
            st.markdown(f"> {item['summary']}", unsafe_allow_html=True)
            for url in item["urls"]:
                clean = clean_url(url)
                st.markdown(
                    f'<div style="margin-bottom:0.5rem;"><a href="{clean}" target="_blank">{clean}</a></div>',
                    unsafe_allow_html=True
                )

except Exception as e:
    # show the full traceback both in the browser and in the terminal
    err = traceback.format_exc()
    st.error("🚨 Unhandled exception in app.py:")
    st.text(err)
    print(err, flush=True)
    # re-raise if you want it in the logs too
    raise