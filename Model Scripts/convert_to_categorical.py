def convert_to_categorical(df, col):
	""" 
	col: string name of the column 
	df: dataframe to be made categorical
	"""
	df2 = df.copy()
	df2['col'] = df2['col'].astype('category')
	cat_columns = df2.select_dtypes(['category']).columns
	df2[cat_columns] = df2[cat_columns].apply(lambda x: x.cat.codes)
	return df2

def populate(df_input, col):
  """ 
  df_input: input dataframe
  col: string name of the column aggregating
  """
  df = df_input.copy() # don't modify original
  
  a = df[col].values.T.tolist()
  b = []
  for item in a:
    for i in range(item):
      b.append(item)
  
  # convert b back into a DataFrame
  df_output = pd.DataFrame(b, columns = list(df.columns.values))
  
  return df_output