from markdown import markdown
import pdfkit

input_filename = input("Whats the filename?\n")
output_filename = input_filename.split(".")[0]+".pdf"

print(f"Thanks!\n\nReading the file {input_filename}...\n") 
with open(input_filename, 'r') as f:
    html_text = markdown(f.read(), output_format='html4')

print(f"Exporting the file as {output_filename}...")
pdfkit.from_string(html_text, output_filename)
print("\nDone!")
