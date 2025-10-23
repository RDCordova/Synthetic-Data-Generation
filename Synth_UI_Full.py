import streamlit as st
import boto3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from botocore.config import Config
from langchain_community.chat_models import BedrockChat
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import time
import zipfile
import pandas as pd
import numpy as np
import math
from io import StringIO
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
import math
from scipy.stats import ks_2samp
from tqdm import tqdm
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from copulas.multivariate import GaussianMultivariate

# Reuse your existing functions from the provided code

# PDF Generation Section
# (Same as in your original code, no changes)

def start_textract_job(bucket_name, document_name):
    textract = boto3.client('textract')
    response = textract.start_document_analysis(
        DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': document_name}},
        FeatureTypes=['TABLES', 'FORMS']
    )
    return response['JobId']

def check_textract_job_status(job_id):
    textract = boto3.client('textract')
    while True:
        response = textract.get_document_analysis(JobId=job_id)
        status = response['JobStatus']
        if status == 'SUCCEEDED':
            return response
        elif status == 'FAILED':
            raise Exception(f"Textract job failed: {response}")
        time.sleep(5)

def extract_text_and_tables_with_layout_from_s3_async(bucket_name, document_name):
    job_id = start_textract_job(bucket_name, document_name)
    response = check_textract_job_status(job_id)
    blocks = response['Blocks']
    text_blocks = []
    table_blocks = []

    def extract_table_data_with_layout(relationship_block, block_map):
        table_data = []
        for relationship in relationship_block['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    cell_block = block_map[child_id]
                    if cell_block['BlockType'] == 'CELL':
                        cell_text = cell_block.get('Text', "")
                        bbox = cell_block['Geometry']['BoundingBox']
                        table_data.append({
                            'text': cell_text,
                            'bbox': bbox
                        })
        return table_data

    block_map = {block['Id']: block for block in blocks}

    for block in blocks:
        if block['BlockType'] == 'LINE':
            bbox = block['Geometry']['BoundingBox']
            text_blocks.append({'text': block['Text'], 'bbox': bbox})
        elif block['BlockType'] == 'TABLE':
            table_data = extract_table_data_with_layout(block, block_map)
            table_blocks.append(table_data)

    return text_blocks, table_blocks

def load_model(top_k, top_p, temperature):
    config = Config(read_timeout=1000)
    bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1', config=config)
    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    model_kwargs = {
        "max_tokens": 100000,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": ["\n\nHuman"],
    }
    model = BedrockChat(client=bedrock_runtime, model_id=model_id, model_kwargs=model_kwargs)
    return model

def generate_new_content_with_mapping(text_blocks, model):
    new_text_blocks = []
    def process_text_block(block):
        prompt = f"Rewrite the following content. Only return the new version, do not include any sort of explanations or prefixes. For example do not include phrases like Here is a new version of the content, preserving the original format :\n\n{block['text']}"
        response = model.invoke(prompt).content
        return {'text': response, 'bbox': block['bbox']}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_text_block, block) for block in text_blocks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating new content", unit="block"):
            new_text_blocks.append(future.result())
    
    return new_text_blocks

def create_pdf_with_mapped_content(mapped_text_blocks, tables, output_file):
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter
    current_y = height - 50
    def scale_bbox_to_page(bbox, width, height):
        x = bbox['Left'] * width
        y = height - (bbox['Top'] * height)
        w = bbox['Width'] * width
        h = bbox['Height'] * height
        return x, y, w, h
    for block in mapped_text_blocks:
        text = block['text']
        bbox = block['bbox']
        x, y, w, h = scale_bbox_to_page(bbox, width, height)
        if current_y - h < 50:
            c.showPage()
            current_y = height - 50
        c.setFont("Helvetica", 10)
        c.drawString(x, current_y, text)
        current_y -= h + 10
    for table in tables:
        table_height = sum([cell['bbox']['Height'] * height for cell in table])
        if current_y - table_height < 50:
            c.showPage()
            current_y = height - 50
        for cell in table:
            cell_text = cell['text']
            bbox = cell['bbox']
            x, y, w, h = scale_bbox_to_page(bbox, width, height)
            if current_y - h < 50:
                c.showPage()
                current_y = height - 50
            c.setFont("Helvetica", 10)
            c.drawString(x, current_y, cell_text)
            c.rect(x, current_y - h, w, h)
            current_y -= h + 10
    c.save()

    
    
def create_zip_from_pdfs(pdf_files, output_zip_file):
    with zipfile.ZipFile(output_zip_file, 'w') as zipf:
        for pdf_file in pdf_files:
            zipf.write(pdf_file, os.path.basename(pdf_file))

def process_pdf_from_s3_with_mapped_content(bucket_name, document_name, output_pdf, top_k, top_p, temperature):
    extracted_text_blocks, table_blocks = extract_text_and_tables_with_layout_from_s3_async(bucket_name, document_name)
    llm = load_model(top_k, top_p, temperature)
    mapped_text_blocks = generate_new_content_with_mapping(extracted_text_blocks, llm)
    create_pdf_with_mapped_content(mapped_text_blocks, table_blocks, output_pdf)

def get_s3_client():
    return boto3.client('s3',region='us-east-1')

# PDF Generation UI
st.set_page_config(page_title="SyntheSys: Content Generation", page_icon="🧊", layout="centered")
st.title("SyntheSys: Content Generation")
st.write("Select the data type to generate (PDF or Tabular) then upload your sample file.\n Once the file is successfully uploaded click the generate button to produce the new output file(s)")
st.markdown("---")

# Step 1: User selects between PDF or Tabular Data Generation
generation_type = st.radio("Select Generation Type", ["Generate PDF(s)", "Generate Tabular Data"])

def get_s3():
    return boto3.client('s3')

def upload_file_to_s3(file, bucket_name, object_name):
    s3 = get_s3()
    try: 
        s3.upload_fileobj(file, bucket_name, object_name)
    except NoCredentialsError:
        st.error("Invaild Credentials")
    except ClientError:
        st.error("Client Error")

# PDF Generation Section
if generation_type == "Generate PDF(s)":
    st.write("### PDF Generation")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    st.sidebar.title("Generation Options")
    num_pdfs = st.sidebar.slider("Number of PDFs to Generate", min_value=1, max_value=10, value=1)
    st.sidebar.header("Adjust PDF Generation Parameters")
    top_k = st.sidebar.slider("Top K (Token Sampling)", min_value=1, max_value=1000, value=250, step=1)
    top_p = st.sidebar.slider("Top P (Probability Threshold)", min_value=0.0, max_value=1.0, value=0.9, step=0.01)
    temperature = st.sidebar.slider("Temperature (Creativity Control)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    if uploaded_file is not None:
        bucket_name = 'a31-cd'
        object_name = uploaded_file.name
        upload_file_to_s3(uploaded_file, bucket_name, object_name)
        st.success("PDF uploaded successfully!")
        st.text_area("Provide addtional information on the desired changes (or leave blank). Your input will be used to aid the generation process.",value="",height=150)
        if st.button("Generate PDFs"):
            with st.spinner("Generating PDF(s)..."):
                def generate_pdfs(num_pdfs):
                    output_files = []
                    for i in range(num_pdfs):
                        output_pdf = f"generated_pdf_{i+1}.pdf"
                        process_pdf_from_s3_with_mapped_content(bucket_name, object_name, output_pdf, top_k, top_p, temperature)
                        output_files.append(output_pdf)
                    return output_files
                generated_pdfs = generate_pdfs(num_pdfs)
                if num_pdfs > 1:
                    zip_name = "generated_pdfs.zip"
                    create_zip_from_pdfs(generated_pdfs,zip_name)
                    st.success(f"PDF(s) generated successfully")
                    with open(zip_name, "rb") as file:
                        st.download_button(f"Download All PDFs as {zip_name}", file, zip_name, mime="application/zip")
                else:
                    pdf_file = generated_pdfs[0]
                    st.success(f"PDF generated successfully")
                    with open(pdf_file, "rb") as file:
                        st.download_button(f"Download PDF", file, pdf_file, mime="application/zip")

# Tabular Data Generation Section (Enhanced)
else:
    st.write("### Tabular Data Generation")
    use_case = st.radio("Select Use Case", ["General (Fastest)", "Modeling", "GenAI"])
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    st.sidebar.title("Generation Options")
    num_obs = st.sidebar.slider("Number of Observations to Generate", min_value=100, max_value=10000, value=1000, step=10)
    top_k = st.sidebar.slider("Top K (Token Sampling)", min_value=1, max_value=1000, value=50, step=1)
    top_p = st.sidebar.slider("Top P (Probability Threshold)", min_value=0.0, max_value=1.0, value=0.9, step=0.01)
    temperature = st.sidebar.slider("Temperature (Creativity Control)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    if uploaded_file is not None:
        bucket_name = 'a31-cd'
        object_name = uploaded_file.name
        upload_file_to_s3(uploaded_file, bucket_name, object_name)
        st.success("CSV uploaded successfully!")
        
        if 'generated_df' not in st.session_state:
            st.session_state['generated_df'] = None
         
        if 'selected_column' not in st.session_state:
            st.session_state['selected_column'] = None
            
        if 'distribution_ckeck' not in st.session_state:
            st.session_state['distribution_ckeck'] = None
        
        
        
        if st.button("Generate Data"):
            with st.spinner("Generating Data..."):
            # Load the uploaded CSV from S3
            
                def read_csv_from_s3(bucket_name, key):
                    s3 = boto3.client('s3')
                    obj = s3.get_object(Bucket=bucket_name, Key=key)
                    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                    return df

                def generate_schema(df):
                    schema = {}

                    # For numeric columns
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_columns:
                        col_stats = {
                            "data_type": "numeric",
                            "mean": df[col].mean(),
                            "std": df[col].std(),
                            "min": df[col].min(),
                            "max": df[col].max(),
                            "25th_percentile": df[col].quantile(0.25),
                            "50th_percentile": df[col].quantile(0.50),  # Median
                            "75th_percentile": df[col].quantile(0.75)
                        }
                        schema[col] = col_stats

                    # For categorical columns
                    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
                    for col in categorical_columns:
                        if df[col].nunique() < 50:  # Treat columns with a small number of unique values as categorical
                            value_counts = df[col].value_counts(normalize=True)
                            col_stats = {
                                "data_type": "categorical",
                                "distribution": value_counts.to_dict()
                            }
                            schema[col] = col_stats

                    # For free-form text columns (treated as 'object' but with high uniqueness)
                    text_columns = df.select_dtypes(include=['object']).columns
                    for col in text_columns:
                        if df[col].nunique() > 50:  # Assume high uniqueness indicates free-form text
                            text_lengths = df[col].dropna().apply(len)  # Calculate the length of each text entry
                            col_stats = {
                                "data_type": "text",
                                "average_length": text_lengths.mean(),
                                "min_length": text_lengths.min(),
                                "max_length": text_lengths.max(),
                                "sample_text": df[col].dropna().sample(3).tolist()  # Provide a few examples
                            }
                            schema[col] = col_stats

                    return schema
                # Load the model
                def load_model(top_k, top_p, temperature):
                    config = Config(read_timeout=1000)
                    bedrock_runtime = boto3.client(service_name='bedrock-runtime', 
                                          region_name='us-east-1',
                                          config=config)

                    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

                    model_kwargs = { 
                        "max_tokens": 200000,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "stop_sequences": ["\n\nHuman"]
                    }

                    model = BedrockChat(
                        client=bedrock_runtime,
                        model_id=model_id,
                        model_kwargs=model_kwargs,
                    )

                    return model
                # Read CSV data and generate synthetic data
                # df = read_csv_from_s3(bucket_name, object_name)
                # schema = generate_schema(df)
                # llm = load_model_for_tabular(top_k, top_p, temperature)

                def format_examples(df):
                    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
                    examples_str = ""
                    for index, row in df[non_numeric_columns].iterrows():
                        row_string = ", ".join([f"{non_numeric_columns[i]}: {row[i]}" for i in range(len(non_numeric_columns))])
                        examples_str += row_string + "\n"
                    return examples_str

                def format_rows(row):
                    return  ", ".join([f"{col}: {row[col]}" for col in row.index if pd.notna(row[col])])

                def row_len(text):
                    return len(text)

                def max_examples(csv_data, max_context_tokens, prompt_tokens, response_token_buffer=0.20):
                    non_numeric_col = csv_data.select_dtypes(exclude=[np.number])
                    formatted_exmaples = non_numeric_col.apply(format_rows, axis =1)
                    char_per_row = int(formatted_exmaples.apply(row_len).mean())
                    max_available = int((max_context_tokens*(1-response_token_buffer)) / char_per_row)
                    max_examples_percent = int(0.6 * len(csv_data))
                    return (min(max_available, max_examples_percent),char_per_row)

                def select_strat_col(csv_data, num_cols=2):
                    cat_cols = csv_data.select_dtypes(include=['object', 'category']).columns
                    score = {}
                    for col in cat_cols:
                        value_counts = csv_data[col].value_counts(normalize=True)
                        max_prop = value_counts.max()
                        score[col] = max_prop
                    sort = sorted(score, key=score.get)
                    return sort[:num_cols]

                def generate_examples(csv_data, num_examples):
                    select_col = select_strat_col(csv_data, num_cols=2)
                    if select_col:
                        sample = csv_data.groupby(select_col, group_keys=False).apply(
                            lambda x: x.sample(frac=min(1.0, num_examples / len(csv_data)))
                        )
                    else:
                        sample = csv_data.sample(num_examples)
                    return sample

                def create_prompt_template_without_numeric():
                    template = """
                    You are a helpful AI assistant for creating new datasets. 
                    Your task is to generate new observations based on the provided examples and schema.
                    The new observations should be similar to the examples.

                    **Categorical Data**: 
                    - For each categorical column, ensure that the frequency distribution of the generated data matches the
                    distribution provided in the schema.
                    - The proportion of each category (e.g., Male/Female) should be preserved based on the frequencies in the schema.
                    - Do no include any null values. Alway use a value from one of the categories.

                    **Free-Form Text**:
                    - For free-form text columns, generate text that is similar in style and structure to the examples provided.
                    - Ensure that the generated text has a length that falls within the range indicated in the schema.
                    - The average length of the generated text should be close to the average length provided in the schema.

                    The formatting of the new observations should match the formatting of the examples: 
                    column 1: value 1, column 2: value 2...column n: value n.

                    **Examples**:
                    {examples}

                    **Schema**:
                    {schema}

                    **Count**: Generate {count} new observations.

                    Only return the new observations, do not include any explanation.
                    """

                    prompt = PromptTemplate(
                        template=template,
                        input_variables=["examples", "schema", "count"]
                    )

                    return prompt

                def generate_response_for_categorical_and_text(csv_data, num_examples, schema, row_tokens):
                    examples = generate_examples(csv_data, num_examples)
                    prompt_examples = format_examples(examples)  # Use only non-numeric examples

                    prompt_template = create_prompt_template_without_numeric()
                    filled_prompt = prompt_template.format(
                        examples=prompt_examples,
                        schema=schema,
                        count=math.ceil((5000 / row_tokens))
                    )

                    message = HumanMessage(content=filled_prompt)
                    response = llm([message])

                    return response.content

                def generate_numeric(original_df, num_rows):
                    if use_case == "General (Fastest)":
                            numeric_columns = original_df.select_dtypes(include=[np.number]).columns
                            sampled_data = pd.DataFrame()
                            for col in numeric_columns:
                                sampled_data[col] = np.random.choice(original_df[col], size=num_rows, replace=True)
                            return sampled_data
                    else:
                        numeric_data = original_df.select_dtypes(include=[np.number])
                        numeric_data = numeric_data.apply(lambda col: col.fillna(col.mean()))
                        copula_model = GaussianMultivariate()
                        copula_model.fit(numeric_data)
                        sampled_data = copula_model.sample(num_rows)

                    return sampled_data

                def create_new_dataframe(text):
                    obs = text.split('\n')
                    data_dicts = []
                    for line in obs:
                        pairs = line.split(', ')
                        entry_dict = {}
                        for pair in pairs:
                            if ': ' in pair:
                                k, v = pair.split(': ', 1)  
                                entry_dict[k.strip()] = v.strip()
                        data_dicts.append(entry_dict)
                    return pd.DataFrame(data_dicts)

                def generate_combined_data(csv_data, num_examples, schema, row_tokens):
                    # Generate numeric
                    numeric_data = generate_numeric(csv_data, num_examples)

                    # Generate categorical and text 
                    response_text = generate_response_for_categorical_and_text(csv_data, num_examples, schema, row_tokens)
                    categorical_and_text_data = create_new_dataframe(response_text)

                    # Combine numeric and categorical/text data
                    combined_data = pd.concat([numeric_data.reset_index(drop=True), categorical_and_text_data.reset_index(drop=True)],
                                              axis=1)

                    # Ensure the combined data has the same column order as the original dataset
                    combined_data = combined_data[csv_data.columns]

                    return combined_data

                def gen_all_obs(gen_cycles, csv_data, num_examples, schema, row_tokens):
                    df_list = []
                    for i in tqdm(range(gen_cycles), desc="Generating Observations"):
                        temp = generate_combined_data(csv_data, num_examples, schema, row_tokens)
                        df_list.append(temp)
                    final_df = pd.concat(df_list, ignore_index=True)
                    return final_df

                def fill_missing_cat(new_data,schema,col):
                    distribution = schema[col]['distribution']
                    cat = list(distribution.keys())
                    probs =  list(distribution.values())

                    missing_idx = new_data[new_data[col].isnull()].index

                    sample_cat = np.random.choice(cat,size=len(missing_idx), p = probs)

                    new_data.loc[missing_idx,col] = sample_cat

                    return new_data



                #run# 
                csv_data = read_csv_from_s3(bucket_name, object_name)
                schema = generate_schema(csv_data)
                llm = load_model(top_k, top_p, temperature)
                max_context_tokens = 20000 
                prompt_tokens = 2000  
                response_token_buffer = 0.2

                num_examples, row_tokens = max_examples(csv_data, max_context_tokens, prompt_tokens, response_token_buffer)

                gen_cycles = math.ceil(num_obs / (20000 / row_tokens))

                obs_df = gen_all_obs(gen_cycles, csv_data, num_examples, schema, row_tokens)

                gen_df = obs_df[:num_obs]

                # Fill missing values in the generated data
                types = dict(csv_data.dtypes)
                for column, dtype in types.items(): 
                    try:
                        gen_df[column] = gen_df[column].astype(dtype)
                    except ValueError as e:

                        print(f"Error converting {column} to {dtype}")

                cat_cols = csv_data.select_dtypes(exclude=[np.number]).columns
                for col in cat_cols:
                    final_df = fill_missing_cat(gen_df,schema,col)

                def convert_to_csv(df):
                    return df.to_csv(index=False)

                def distribution_check_summary(original_data,new_data, p_value_threshold=0.05):
                    results = []
                    num_cols = original_data.select_dtypes(include=[np.number]).columns
                    cat_cols = original_data.select_dtypes(exclude=[np.number]).columns

                    for col in num_cols: 
                        ks_stat, p_value = ks_2samp(original_data[col], new_data[col])
                        result = "Pass" if p_value > p_value_threshold else "Fail"

                        results.append({'Column': col,
                                        'Type': 'Numeric',
                                        'P-Value': p_value,
                                        'Test': 'KS Test',
                                        'Result': result
                                       }
                            )

                    for col in cat_cols: 
                        original_count = original_data[col].value_counts()
                        new_count = new_data[col].value_counts()

                        counts = pd.DataFrame({'original': original_count, 'generated': new_count}).fillna(0)
                        chi2_stat, p_value, dof, ex = chi2_contingency(counts.T)
                        result = "Pass" if p_value > p_value_threshold else "Fail"

                        results.append({'Column': col,
                                        'Type': 'Categorical',
                                        'P-Value': p_value,
                                        'Test': 'Chi-Square Test',
                                        'Result': result
                                       }
                            ) 
                    result_df = pd.DataFrame(results)
                    return result_df

                st.session_state['distribution_check'] = distribution_check_summary(csv_data,final_df)
                st.session_state['generated_df'] = final_df

                csv_download = convert_to_csv(final_df)
                st.success(f"Data generated successfully")
                st.download_button("Download Generated Data", csv_download, "generated_data.csv", mime="text/csv")
        


        if st.session_state['generated_df'] is not None:
            generate_df = st.session_state['generated_df']
            
            def read_csv_from_s3(bucket_name, key):
                s3 = boto3.client('s3')
                obj = s3.get_object(Bucket=bucket_name, Key=key)
                df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                return df 
            csv_data = read_csv_from_s3(bucket_name,object_name)


            if st.session_state['distribution_check'] is not None:
                st.write("### Quality Check")
                st.write("All tests are evaluated against a p-value of 0.05. A p-value greater than 0.05 is a pass.\nOnly numeric and categorical values columns are checked.")
                st.dataframe(st.session_state['distribution_check'])

            selected_column = st.selectbox("Select a column to plot the distrbution:", csv_data.columns)

            st.session_state['selected_column'] = selected_column

            if selected_column:
                fig, ax = plt.subplots(figsize=(10,6))

                if pd.api.types.is_numeric_dtype(csv_data[selected_column]):
                    sns.kdeplot(csv_data[selected_column], color='blue', label = 'Original', ax=ax, fill=True)
                    sns.kdeplot(generate_df[selected_column], color='green', label = 'Generated', ax=ax, fill=True)
                    ax.set_xlabel(selected_column)
                    ax.set_ylabel("Density")
                    ax.set_title(f"Distribution comparison {selected_column}: Original vs Generated")
                    ax.legend()
                else:
                    original_counts = csv_data[selected_column].value_counts(normalize=True)
                    generated_counts = generate_df[selected_column].value_counts(normalize=True)
                    
                    combined_df = pd.DataFrame({
                        'Category': original_counts.index, 
                        'Original': generated_counts.values,
                        'Generated': generated_counts.reindex(original_counts.index).fillna(0).values
                    })
                    
                    melted_df = pd.melt(combined_df, id_vars='Category', value_vars=['Original','Generated'],
                                        var_name='Dataset', value_name='Proportion')
                                        
                    sns.barplot(x='Category', y='Proportion', hue='Dataset', data= melted_df, ax=ax, palette=['blue','green'])
                    ax.set_xlabel(selected_column)
                    ax.set_ylabel('Proportion')
                    ax.set_title(f"Distribution comparison {selected_column}: Original vs Generated")
                    
                st.pyplot(fig)




            
