import asyncio
import os
import time

from openai import OpenAI
from dotenv import load_dotenv

from utils import create_full_version, html_to_pdf, read_files, write_markdown_as_html, create_summary

load_dotenv()

folder = "files"

output_filename = os.path.join(folder, "output.pdf")
output_summary_filename = os.path.join(folder, "output-summary.pdf")
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

start_time = time.time()
# ------------------------ STEP 1 ------------------------

print("------------------------ STEP 1 ------------------------")
print("Parsing PDF files...")
files_content = read_files(folder)
print("PDF files successfully parsed")

# ------------------------ STEP 2 ------------------------

print("------------------------ STEP 2 ------------------------")
print("Creating full version...")
full_version = create_full_version(files_content, client)
print("Full version successfully created")

print("Generating HTML...")
html_path = write_markdown_as_html(full_version)
print(f"HTML successfully saved to {html_path}")

print("Generating PDF...")
asyncio.run(html_to_pdf(html_path, output_filename))
print("PDF successfully saved")

# ------------------------ STEP 3 ------------------------

print("------------------------ STEP 3 ------------------------")
print("Creating summary...")
summary = create_summary(files_content, full_version, client)
print("Summary successfully created")

print("Generating HTML...")
html_path = write_markdown_as_html(summary)
print(f"HTML successfully saved to {html_path}")


print("Generating PDF...")
asyncio.run(html_to_pdf(html_path, output_summary_filename))
print("PDF successfully saved")
print(f"Done script in {time.time() - start_time}")
