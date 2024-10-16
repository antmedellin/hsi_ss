import shutil

# path of the folder to be zipped

folder_path='submitted_results/submission3/results'

# name of the output zip file(without.zip extension)

output_zip_path='submitted_results/submission3/results'

shutil.make_archive(output_zip_path, 'zip', folder_path)

print(f"Successfully created {output_zip_path}.zip")