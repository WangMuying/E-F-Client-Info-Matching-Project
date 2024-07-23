import pandas as pd
from sentence_transformers import SentenceTransformer, util # Run this line in the terminal to install the necessary packages: pip install -U sentence-transformers
import torch
from torch.utils.data import DataLoader, Dataset


def read_data(file_path, encodings=['latin1', 'iso-8859-1', 'cp1252']):
    '''
    Read a CSV file with the specified encodings.

    Parameters:
        file_path (str): The path to the CSV file.
        encodings (list): A list of encodings to try when reading the CSV file.
    
    Returns:
        df (pd.DataFrame): The DataFrame containing the CSV data.
    '''

    # Try reading the CSV with a different encoding
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully read the CSV file {file_path} with {encoding} encoding.")
            break
        except UnicodeDecodeError:
            print(f"Failed to read the CSV file {file_path} with {encoding} encoding.")
    else:
        print("Unable to read the CSV file with the specified encodings.")
    return df


def make_composite_key(df_conference, df_all, event_composite_key_list, salesforce_composite_key_list):
    '''
    Make composite keys for conference data (df_conference) and salesforce data (df_all).

    Compostie Key = Full Name + ' ' + Institution

    Parameters:
        df_conference (pd.DataFrame): The DataFrame containing the conference data.
        df_all (pd.DataFrame): The DataFrame containing the salesforce data.
        event_composite_key_list (list): A list of field names to use for the composite key in the conference data.
        salesforce_composite_key_list (list): A list of field names to use for the composite key in the salesforce data.

    Returns:
        attendees_df_llm (pd.DataFrame): The DataFrame containing the conference data with the composite key.
        customers_df_llm (pd.DataFrame): The DataFrame containing the salesforce data with the composite key.
    '''

    attendees_df_llm = df_conference.copy() # attendees data
    customers_df_llm = df_all.copy()  # salesforce data

    # Fill na values with empty strings to make sure the Encoding model works
    # attendees_df_llm['First Name'] = attendees_df_llm['First Name'].fillna('').astype(str)
    # attendees_df_llm['Last Name'] = attendees_df_llm['Last Name'].fillna('').astype(str)
    # attendees_df_llm['Institution'] = attendees_df_llm['Institution'].fillna('').astype(str)
    for key in event_composite_key_list:
        attendees_df_llm[key] = attendees_df_llm[key].fillna('').astype(str)

    for key in salesforce_composite_key_list:
        customers_df_llm[key] = customers_df_llm[key].fillna('').astype(str)

    # Create Composite Keys
    # attendees_df_llm['Composite Key'] = attendees_df_llm['First Name'] + ' ' + attendees_df_llm['Last Name'] + ' ' + attendees_df_llm['Institution']
    attendees_df_llm['Composite Key'] = attendees_df_llm.apply(lambda row: ' '.join([row[key] for key in event_composite_key_list]), axis=1)
    customers_df_llm['Composite Key'] = customers_df_llm.apply(lambda row: ' '.join([row[key] for key in salesforce_composite_key_list]), axis=1)
    return (attendees_df_llm, customers_df_llm)


class TextDataset(Dataset):
    '''A custom dataset class for the SentenceTransformer model.'''

    # Initialize the dataset
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def compute_embeddings(model, device, text_list, batch_size=32):
    '''
    Compute embeddings for a list of texts using the SentenceTransformer model.

    Parameters:
        model (SentenceTransformer): The SentenceTransformer model to use for encoding.
        device (str): The device to use for encoding ('cuda' or 'cpu').
        text_list (list): A list of texts to encode.
        batch_size (int): The batch size to use for encoding in the purpose of reducing memory allocation.
    
    Returns:
        embeddings (torch.Tensor): The embeddings for the list of texts.
    '''
    
    dataset = TextDataset(text_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    for batch in dataloader:
        batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False, device=device)
        embeddings.append(batch_embeddings)

    return torch.cat(embeddings)


# Function that pass keys to compute_embeddings() function
def get_embeddings(model, device, 
                   attendees_df_llm, customers_df_llm, 
                   event_name_key_list, event_institution_key_list, 
                   salesforce_name_key_list, salesforce_institution_key_list, 
                   batch_size=32):
    '''
    Get embeddings (vectors) for the Composite Key, Name, and Institution for the conference data and salesforce data.
    
    Parameters:
        model (SentenceTransformer): The SentenceTransformer model to use for encoding.
        device (str): The device to use for encoding ('cuda' or 'cpu').
        attendees_df_llm (pd.DataFrame): The DataFrame containing the conference data with the composite key.
        customers_df_llm (pd.DataFrame): The DataFrame containing the salesforce data with the composite key.
        event_name_key_list (list): A list of field names to use for the client full name in the conference data.
        event_institution_key_list (list): A list of field names to use for the institution name in the conference data.
        salesforce_name_key_list (list): A list of field names to use for the client full name in the salesforce data.
        salesforce_institution_key_list (list): A list of field names to use for the institution name in the salesforce data.
        batch_size (int): The batch size to use for encoding in the purpose of reducing memory allocation.
    
    Returns:
        attendees_composite_embeddings (torch.Tensor): The embeddings for the Composite Key in the conference data.
        customers_composite_embeddings (torch.Tensor): The embeddings for the Composite Key in the salesforce data.
        attendees_name_embeddings (torch.Tensor): The embeddings for the Name in the conference data.
        customers_name_embeddings (torch.Tensor): The embeddings for the Name in the salesforce data.
        attendees_institution_embeddings (torch.Tensor): The embeddings for the Institution in the conference data.
        customers_account_embeddings (torch.Tensor): The embeddings for the Institution in the salesforce data.
    '''

    # Get embeddings for the Composite Key, Name, Institution respectively
    attendees_composite_embeddings = compute_embeddings(model, device, attendees_df_llm['Composite Key'].tolist(), batch_size)
    customers_composite_embeddings = compute_embeddings(model, device, customers_df_llm['Composite Key'].tolist(), batch_size)

    attendees_name_embeddings = compute_embeddings(model, device, attendees_df_llm.apply(lambda row: ' '.join([row[key] for key in event_name_key_list]), axis=1).tolist(), batch_size)
    customers_name_embeddings = compute_embeddings(model, device, customers_df_llm.apply(lambda row: ' '.join([row[key] for key in salesforce_name_key_list]), axis=1).tolist(), batch_size)

    attendees_institution_embeddings = compute_embeddings(model, device, attendees_df_llm.apply(lambda row: ' '.join([row[key] for key in event_institution_key_list]), axis=1).tolist(), batch_size)
    customers_account_embeddings = compute_embeddings(model, device, customers_df_llm.apply(lambda row: ' '.join([row[key] for key in salesforce_institution_key_list]), axis=1).tolist(), batch_size)

    return (attendees_composite_embeddings, customers_composite_embeddings, attendees_name_embeddings, customers_name_embeddings, attendees_institution_embeddings, customers_account_embeddings)


def match_records(attendees_composite_embeddings, customers_composite_embeddings, 
                  attendees_name_embeddings, customers_name_embeddings, 
                  attendees_institution_embeddings, customers_account_embeddings, 
                  attendees_df_llm, customers_df_llm, 
                  lower_bound_threshold=0.50, threshold=0.90):
    '''
    Match the conference data with the salesforce data based on the embeddings cosine similarity.

    If the cosine similarity scores for the Composite Key, Name, AND Institution are ALL above the threshold (default is 0.90), then it is considered a MATCHED record.
    If the cosine similarity scores for the Composite Key, Name, OR Institution are below the lower bound threshold (default is 0.50), then it is considered an UNMATCHED record.
    Otherwise, it is considered a REVIEW record.

    Parameters:
        attendees_composite_embeddings (torch.Tensor): The embeddings for the Composite Key in the conference data.
        customers_composite_embeddings (torch.Tensor): The embeddings for the Composite Key in the salesforce data.
        attendees_name_embeddings (torch.Tensor): The embeddings for the Name in the conference data.
        customers_name_embeddings (torch.Tensor): The embeddings for the Name in the salesforce data.
        attendees_institution_embeddings (torch.Tensor): The embeddings for the Institution in the conference data.
        customers_account_embeddings (torch.Tensor): The embeddings for the Institution in the salesforce data.
        attendees_df_llm (pd.DataFrame): The DataFrame containing the conference data with the composite key.
        customers_df_llm (pd.DataFrame): The DataFrame containing the salesforce data with the composite key.
        lower_bound_threshold (float): The lower bound threshold for the cosine similarity scores.
        threshold (float): The upper bound threshold for the cosine similarity scores.

    Returns:
        results_df (pd.DataFrame): The DataFrame containing the results of the matching process.
    '''
    
    results = []
    for i, attendee_embedding in enumerate(attendees_composite_embeddings):

        # Calculate the cosine similarity scores for each composite key
        composite_scores = util.pytorch_cos_sim(attendee_embedding, customers_composite_embeddings)
        max_composite_score, max_composite_idx = torch.max(composite_scores, dim=1)

        # Caclulate the cosine similarity scores for the name
        name_scores = util.pytorch_cos_sim(attendees_name_embeddings[i], customers_name_embeddings)
        max_name_score = name_scores[0, max_composite_idx].item()

        # Calculate the cosine similarity scores for the institution
        institution_scores = util.pytorch_cos_sim(attendees_institution_embeddings[i], customers_account_embeddings)
        max_institution_score = institution_scores[0, max_composite_idx].item()

        # If ANY of the scores are below the lower bound threshold, then it is considered an unmatched record
        if max_composite_score.item() < lower_bound_threshold or max_name_score < lower_bound_threshold or max_institution_score < lower_bound_threshold:
            results.append((attendees_df_llm.iloc[i]['Composite Key'], None, max_composite_score.item(), max_name_score, max_institution_score, "Unmatched"))
        # If ALL of the scores are above the threshold, then it is considered a matched record
        elif max_composite_score.item() >= threshold and max_name_score >= threshold and max_institution_score >= threshold:
            best_match = customers_df_llm.iloc[max_composite_idx.item()]['Composite Key']
            results.append((attendees_df_llm.iloc[i]['Composite Key'], best_match, max_composite_score.item(), max_name_score, max_institution_score, "Matched"))
        # If ANY of the scores are below the review threshold, then it is considered a review record
        else:
            best_match = customers_df_llm.iloc[max_composite_idx.item()]['Composite Key']
            results.append((attendees_df_llm.iloc[i]['Composite Key'], best_match, max_composite_score.item(), max_name_score, max_institution_score, "Review"))

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results, columns=['Attendee Composite Key', 'Matched Customer Composite Key', 'Composite Similarity Score', 'Name Similarity Score', 'Institution Similarity Score', 'Status'])
    return results_df


def update_id(row, salesforce_id):
    '''
    Update the Salesforce ID based on the status of the record.

    If the record is matched or reviewed to be 1 (reviewed to be matched), then update the Salesforce ID with the matched customer's Salesforce ID.
    Otherwise, keep the Salesforce ID as is (the customer's Salesforce ID for unmatched records are empty, and records reviewed to be 0 will be left empty).

    Parameters:
        row (pd.Series): The row of the DataFrame.
        salesforce_id (str): The field name of the Salesforce ID in the DataFrame.
        
    Returns:
        row[salesforce_id] (str): The updated Salesforce ID.
    '''

    # if (row['Status'] == 'Matched') | (row['Review'] == 1) : # matched & unmatched (NaN) or reviewed to be matched (1)
    if row['Review'] != 0:
        return row[salesforce_id + ' Customer']
    return


def get_model():
  ''' Get the SentenceTransformer model for encoding the text data. '''
  # Load Model
  model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
  return model


def get_matching_output(model, device, 
                        event_name, event_data_path, event_feature_dict, 
                        salesforce_data_date, salesforce_client_type_list, salesforce_feature_dict_list, ind):
  '''
  Get the matching output for the conference data and salesforce data.
  
  Parameters:
      model (SentenceTransformer): The SentenceTransformer model to use for encoding.
      device (str): The device to use for encoding ('cuda' or 'cpu').
      event_name (str): The name of the event.
      event_data_path (str): The path to the conference data.
      event_feature_dict (dict): The dictionary containing the features of the conference data, e.g. field names for composite key, name, and institution.
      salesforce_data_date (str): The export date of the salesforce data.
      salesforce_client_type_list (list): A list of client types in the salesforce data, e.g. ['contacts', 'leads', 'unidentified_leads'].
      salesforce_feature_dict_list (list): A list of dictionaries containing the features of the salesforce data for each client type.
      ind (int): The index of the salesforce client type to match with the conference. 0 for the first client type, 1 for the second client type, etc.

  Returns:
      None
  '''
  
  salesforce_client_type = salesforce_client_type_list[ind]
  salesforce_feature_dict = salesforce_feature_dict_list[ind]

  print('-'*100)
  print(f'Process starts! Matching {event_name} data with Salesforce all_{salesforce_client_type} data...')
  print('-'*100)

  # Step 1. Read data
  salesforce_data_path = f'{salesforce_data_date}_all_{salesforce_client_type}_salesforce.csv'
  df_salesforce = read_data(salesforce_data_path)

  # First, match the conference data with all_contacts data (the index for contacts client type is 0)
  # If already matched with contacts, then match the rest of the records in the conference data with leads (ind=1) and unidentified leads (ind=2) data, respectively.
  if ind > 0:
    event_data_path = f'{event_name}_unmatched_with_{salesforce_client_type_list[ind-1]}.csv'
  unmatched_event_data = read_data(event_data_path)

  # Get the features for the conference data and salesforce data
  event_composite_key_list = event_feature_dict.get('event_composite_key_list', [])
  event_name_key_list = event_feature_dict.get('event_name_key_list', [])
  event_institution_key_list = event_feature_dict.get('event_institution_key_list', [])

  all_composite_key_list = salesforce_feature_dict.get('salesforce_composite_key_list', [])
  all_name_key_list = salesforce_feature_dict.get('salesforce_name_key_list', [])
  all_institution_key_list = salesforce_feature_dict.get('salesforce_institution_key_list', [])
  salesforce_id = salesforce_feature_dict.get('salesforce_id', '')

  # Step 2. Make composite keys
  attendees_df_llm, customers_df_llm = make_composite_key(unmatched_event_data, df_salesforce, event_composite_key_list, all_composite_key_list)

  # Step 3. Get embeddings for the composite keys, names, and institutions respectively
  # This should take approx 10-15 mins if gpu else 30+ mins.
  attendees_composite_embeddings, customers_composite_embeddings, \
    attendees_name_embeddings, customers_name_embeddings, \
    attendees_institution_embeddings, customers_account_embeddings = get_embeddings(model, device, 
                                                                                    attendees_df_llm, customers_df_llm, 
                                                                                    event_name_key_list, event_institution_key_list, 
                                                                                    all_name_key_list, all_institution_key_list)

  # Step 4. Match records
  results_df = match_records(attendees_composite_embeddings, customers_composite_embeddings, 
                             attendees_name_embeddings, customers_name_embeddings, 
                             attendees_institution_embeddings, customers_account_embeddings, 
                             attendees_df_llm, customers_df_llm, 
                             lower_bound_threshold=0.5, threshold=0.90)
  
  assert len(results_df) == len(attendees_df_llm), "Length of results_df doesn't match length of attendees_df_llm. There may be duplicates in each data frame."
  results_df.drop_duplicates(subset=['Attendee Composite Key'], keep='first', inplace=True)
  attendees_df_llm.drop_duplicates(subset=['Composite Key'], keep='first', inplace=True)
  assert len(results_df) == len(attendees_df_llm), "Lengths still unmatched after dropping duplicates."


  # Step 5. Merge the results with the event data to get the attendees' information
  output = results_df.merge(attendees_df_llm, how="left", left_on='Attendee Composite Key', right_on='Composite Key')
  # Drop duplicates
  if output.duplicated(subset=['Attendee Composite Key']).sum() > 0:
      output.drop_duplicates(subset=['Attendee Composite Key'], keep='first', inplace=True)
  assert len(results_df) == len(output), f"Merging back {event_name} (attendees) data goes wrong."
  output = output.drop(columns=['Composite Key'])
  ## Delete Salesforce column later for other conference data. Here, Salesforce is just a (assume true) label for accuracy checking
  # output = output.rename(columns={'First Name': 'First Name Attendee', 'Last Name': 'Last Name Attendee', 'Institution': 'Institution Attendee', 'Salesforce': 'Salesforce Attendee'})
  # output = output.rename(columns={'First Name': 'First Name Attendee', 'Last Name': 'Last Name Attendee', 'Institution': 'Institution Attendee'})
  output = output.rename(columns={key: key + ' Attendee' for key in event_composite_key_list}) # label the columns from the conference data with 'Attendee' in the end
  len_after_merge_1 = len(output)

  # Step 6. Merge the results with the salesforce data to get potential salesforce_id
  output = output.merge(customers_df_llm, how="left", left_on='Matched Customer Composite Key', right_on='Composite Key') # Salesforce ID comes from this table
  # Drop duplicates
  if output.duplicated(subset=['Attendee Composite Key']).sum() > 0:
      output.drop_duplicates(subset=['Attendee Composite Key'], keep='first', inplace=True)
  assert len_after_merge_1 == len(output), "Merging back Salesforce data goes wrong."

  # Drop all the columns in customers_df_llm except for salesforce_id
  output = output.drop(columns=['Composite Key'])
  output = output.drop(columns=all_composite_key_list)
  output = output.rename(columns={salesforce_id: salesforce_id + ' Customer'})

  cols_to_keep = results_df.columns.tolist()
  cols_to_keep.extend([key + ' Attendee' for key in event_composite_key_list])
  cols_to_keep.extend([salesforce_id + ' Customer'])

  output = output[cols_to_keep]
  output.to_excel(f'output_all_{salesforce_client_type}.xlsx', index=False)

  # Step 7. Manual review
  while True:
    print('')
    print(f'''
    ----------------------------------------------------------------------------
    output_all_{salesforce_client_type}.xlsx is saved.
    ----------------------------------------------------------------------------
    1. Make a copy of `output_all_{salesforce_client_type}.xlsx` and open it.
    2. Add a new column called `Review`.
    3. Filter out the entries with `Status` being `Review`
    4. Fill 1 if reviewed to be matched, 0 if unmatched.
    5. Save the file as `output_all_{salesforce_client_type}_after_review.xlsx`.
    ----------------------------------------------------------------------------
    ''' )

    status = input("Review done? (y/n): ")
    if status.lower() == 'y':
      break
    else:
      continue
  
  # Step 8. Update the Salesforce ID based on the review
  output_after_review = pd.read_excel(f'output_all_{salesforce_client_type}_after_review.xlsx')
  output_after_review[salesforce_id + ' Customer'] = output_after_review.apply(update_id, salesforce_id=salesforce_id, axis=1)

  # Step 9. Save the unmatched records to a new CSV file for further matching
  rest_of_event = output_after_review[output_after_review[salesforce_id + ' Customer'].isnull()][[key + ' Attendee' for key in event_composite_key_list]]
  # rest_of_enf.rename(columns={'First Name Attendee': "First Name", 'Last Name Attendee': "Last Name", 'Institution Attendee': "Institution"}, inplace=True)
  rest_of_event.rename(columns={key + ' Attendee': key for key in event_composite_key_list}, inplace=True)
  rest_of_event.to_csv(f'{event_name}_unmatched_with_{salesforce_client_type}.csv', index=False)
  print('----------------------------------------------------------------------------')
  print(f'{event_name}_unmatched_with_{salesforce_client_type}.csv is saved.')

  # Step 10. Save the matched records to a new CSV file with only ids for uploading
  ready_for_upload = output_after_review[[salesforce_id + ' Customer']].rename(columns={salesforce_id + ' Customer': salesforce_id})
  # ready_for_upload = ready_for_upload.rename(columns={'First Name Attendee': "First Name", 'Last Name Attendee': "Last Name", 'Institution Attendee': "Institution",'Contact ID Customer': 'Contact ID'})
  ready_for_upload = ready_for_upload[~ready_for_upload[salesforce_id].isnull()]
  ready_for_upload.to_csv(f'ready_for_upload_all_{salesforce_client_type}.csv', index=False)

  print(f'ready_for_upload_all_{salesforce_client_type}.csv is saved.')
  print('----------------------------------------------------------------------------')
  print('Matching finished.')
  return


if __name__ == '__main__':
  '''Main function to run the matching process.'''

  # Conference specific path
  event_name = 'private_wealth_gp'
  event_data_path = 'Annual_Private_Wealth_Great_Plains_Forum.csv'

  # Conference specific field names.
  event_feature_dict = {
      'event_composite_key_list': ['Name', 'Firm'],
      'event_name_key_list': ['Name'],
      'event_institution_key_list': ['Firm']
  }

  # Salesforce data export date.
  salesforce_data_date = '2024-07-11'

  # Salesforce data field. No need to change if there's no change in Salesforce objects.
  salesforce_client_type_list = ['contacts', 'leads', 'unidentified_leads']
  salesforce_feature_dict_list = [{'salesforce_client_type': 'contacts',
    'salesforce_composite_key_list': ['First Name', 'Last Name', 'Account Name'],
    'salesforce_name_key_list': ['First Name', 'Last Name'],
    'salesforce_institution_key_list': ['Account Name'],
    'salesforce_id': 'Contact ID'
    },

  {'salesforce_client_type': 'leads',
    'salesforce_composite_key_list': ['First Name', 'Last Name', 'Account'],
    'salesforce_name_key_list': ['First Name', 'Last Name'],
    'salesforce_institution_key_list': ['Account'],
    'salesforce_id': 'Lead ID'},

  {'salesforce_client_type': 'unidentified_leads',
    'salesforce_composite_key_list': ['Unidentified Lead: Unidentified Leads', 'Firm Name'],
    'salesforce_name_key_list': ['Unidentified Lead: Unidentified Leads'],
    'salesforce_institution_key_list': ['Firm Name'],
    'salesforce_id': 'Unidentified Lead: ID'}
  ]

  # Check availability for gpu
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use this line if running in colab and gpu is available
  device = 'mps' if torch.backends.mps.is_available() else 'cpu' # use this line if running in a mac book with mps
  print('-'*100)
  print("Device using is", device)
  print('-'*100)

  # Get pre-trained model
  model = get_model()
  model.to(device) # Move model to GPU if available

  # Change the date for the data file inside the function get_matching_output
  for ind in range(1, len(salesforce_client_type_list)-1):
    get_matching_output(model, device, event_name, event_data_path, event_feature_dict, salesforce_data_date, salesforce_client_type_list, salesforce_feature_dict_list, ind)