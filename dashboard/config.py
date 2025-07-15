basedir = "/home/akash_p23897/"

CONFIG = {
    "data_file": basedir+"summarizer_module/articles_export.csv",
    "summaries_file": basedir+"test_data/data_page_Canada_Feb21.json",
    "country_origin_file": basedir+"dashboard/website_country_mapping.json",
    "TIME_SCALE": "Month",  # Options: "Day", "Month", "Year"
    "keywords" : ["Canada"],
    "vertex_project": "your-gcp-project-id",
    "vertex_region":  "us-central1",            
    "summarizer_endpoint": "projects/…/locations/…/endpoints/1234567890",  

}


