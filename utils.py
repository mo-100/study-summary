from pathlib import Path
import markdown
from openai import OpenAI
from pypdf import PdfReader
from playwright.async_api import async_playwright

import os
import tempfile

# ---------------------- CONSTANTS ----------------------

TEMPERATURE = 1
TOP_P = 0.95
REASONING = "high"
MODEL = "gemini-2.5-flash-preview-04-17"

# ---------------------- READ FILES ----------------------


def read_files(folder: str) -> list[str]:
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    # ensure sorted files
    print(f"Found files: {pdf_files}")
    files_content = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder, pdf_file)
        reader = PdfReader(pdf_path)
        text = [f"PAGE {page.page_number}:\n{page.extract_text()}" for page in reader.pages]
        full_text = "\n\n".join(text)
        full_text_with_name = f"{pdf_file} File Contents: {full_text}"
        files_content.append(full_text_with_name)
    return files_content


# ---------------------- API CALLS ----------------------

FULL_PROMPT = """I provided you with some files and references. I want you to deeply analyze them and generate a comprehensive, highly structured breakdown in a clear, progressive step-by-step format. The goal is to make this breakdown fully understandable, memorable, and usable even by someone unfamiliar with the original source.

Follow these instructions exactly:
	1. First, extract the core purpose and scope of the data provided. Summarize it in 2-3 sentences, in plain, simple language.
	2. Next, break down the structure into main sections or parts, each with a short descriptive title and a clear explanation. Use hierarchy and indentation to reflect logical organization (like an outline).
	3. For each section, explain it step-by-step. Go in a teaching style, as if you’re guiding a student through the concept. Use:
        - Simple analogies and real-life comparisons
        - Visual metaphors (e.g., "think of it like a flow of water…")
        - Diagrams or bullet steps when needed (you can describe the diagrams if not able to draw)
	4. Use chunking for memory: Group related information into logical, bite-sized chunks with headers, short explanations, and mnemonic aids (such as acronyms or storytelling).
	5. At the end of each section, give a mini-summary + quick review question or mental quiz (e.g., "Can you recall what step 3 does and why it matters?").
	6. Finally, provide a summary recap of everything covered, in list format, including:
        - The logical flow from start to end
        - Important key points to remember
        - A metaphor or image to keep in mind that helps tie it all together

Focus on clarity, depth, and flow — make the explanation so well-structured and vivid that it naturally sticks in the reader’s memory."""


def create_full_version(contents: list[str], client: OpenAI) -> str:
    messages = [{"role": "system", "content": FULL_PROMPT}]
    pdfs = [{"role": "user", "content": c} for c in contents]
    messages.extend(pdfs)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        reasoning_effort=REASONING,
    )
    return response.choices[0].message.content


SUMMARY_PROMPT = """You received a set of files covering multiple lectures. Create a comprehensive, exam-focused study guide that is also easy to understand, memorize, and apply. It must balance ultra-concise rapid recall sections with clear step-by-step understanding of the deeper logic behind each part.

Structure the output using Markdown with the following rules:

Goal:
Help the student fully understand, memorize, and recall the material under exam pressure — even if they’re skimming before the test.

Format Instructions:
    1. Start with a 2-line summary of the topic’s core purpose.
    2. Break it into sections per lecture or theme, using headers like:
        - Lecture 1: Topic Title – Quick View + Breakdown
    3. Inside each section:
        - Begin with a "Quick Recall" block of key terms or definitions.
        - Follow with a step-by-step mini-guide, using bullets or numbers.
        - Use real-world analogies, when possible, to explain abstract parts.
        - Use tables to compare or contrast related ideas.
        - End the section with a 1-line mental quiz or recall question.
    4. After all lectures, include:
        - Common Pitfalls — list top 3–5 mistakes to avoid in the exam.
        - Exact Terminology — list words or phrases required for full marks.
    5. End with:
        - Recap: One-Minute Summary — give the full flow in ~5–7 bullets.
        - Visual Memory Aid — describe one strong metaphor, shape, or diagram that helps tie the material together.
        - Last-Minute Study Tactics — 2–3 actionable tips using the cheat sheet (e.g., "cover the Quick Recall and quiz yourself," etc.)

Constraints:
    - Prioritize clarity, compactness, and logic flow.
    - Keep explanations short, structured, and memory-friendly.
    - No fluff, no intros — just exam-ready material with teaching clarity.
    - Provide the definitions to help memorizing them."""


def create_summary(contents: list[str], full_version: str, client: OpenAI) -> str:
    messages = [{"role": "system", "content": FULL_PROMPT}]
    pdfs = [{"role": "user", "content": c} for c in contents]
    messages.extend(pdfs)
    messages.append({"role": "assistant", "content": full_version})
    messages.append({"role": "user", "content": SUMMARY_PROMPT})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        reasoning_effort=REASONING,
    )
    return response.choices[0].message.content


# ---------------------- MARKDOWN TO HTML ----------------------


def write_markdown_as_html(markdown_content: str) -> str:
    html_content = markdown.markdown(markdown_content, extensions=["markdown.extensions.tables"])
    full_html_content = (
        """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Study Summarizer</title>
        <style>
            body {
                /* --- Font Settings --- */
                font-family: Optima, Candara, Calibri, Segoe, "Segoe UI", Optima, Arial, sans-serif;

                /* --- Background Settings --- */
                /* Keep background on body too - good practice */
                background-color: #f0f0f0;

                /* --- Spacing --- */
                /* Removed margin and added padding. */
                /* Padding adds space *inside* the element's background area. */
                /* This creates space between the content and the edges of the body's background. */
                padding: 20px; /* Add 20px spacing inside the body */
                margin: 0; /* Remove default browser margin */

                /* --- Layout --- */
                /* Ensures body covers at least the full viewport height. */
                /* With padding, this makes the *content area* + *padding* equal 100vh. */
                min-height: 100vh;

                /* Optional: To prevent horizontal scrollbar if padding + content exceeds viewport width */
                box-sizing: border-box;
            }

            /* --- Table Styling --- */
            /* Add borders to tables, table headers, and table data cells */
            table, th, td {
                border: 1px solid black; /* 1px solid black border */
            }

            /* Collapse borders so they don't create double lines */
            table {
                border-collapse: collapse;
                width: 100%; /* Optional: Make tables span the container width */
                margin-top: 10px; /* Optional: Add some space above tables */
                margin-bottom: 10px; /* Optional: Add some space below tables */
            }

            /* Optional: Add some padding inside table cells */
            th, td {
                padding: 8px;
                text-align: left; /* Optional: Align text left */
            }
        </style>
    </head>
    <body>"""
        + html_content
        + """</body>
    </html>
    """
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as temp_file:
        # Write the full HTML content to the temporary file
        temp_file.write(full_html_content)
        # The file is automatically closed when exiting the 'with' block

    # Return the path of the temporary file
    return temp_file.name


# ---------------------- HTML TO PDF ----------------------


async def html_to_pdf(html_file_path: str, pdf_output_path: str):
    html_path_obj = Path(html_file_path).resolve()  # Get absolute path
    pdf_path_obj = Path(pdf_output_path).resolve()

    # Ensure the input HTML file exists
    if not html_path_obj.is_file():
        print(f"Error: HTML file not found at {html_path_obj}")
        return

    # Ensure the output directory exists (optional, creates if not)
    pdf_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Convert the file path to a file:/// URL
    # This is crucial for Playwright to open local files
    file_url = html_path_obj.as_uri()
    print(f"Attempting to open: {file_url}")

    async with async_playwright() as p:
        # Launch the browser (Chromium is generally good for PDFs)
        # You can also use p.firefox or p.webkit
        browser = await p.chromium.launch()
        page = await browser.new_page()

        try:
            # Navigate to the local HTML file using the file:// URL
            # 'load' is usually sufficient for static local files
            # 'networkidle' waits until network connections are idle (more relevant for web pages)
            await page.goto(file_url, wait_until="load")  # or 'networkidle' or 'domcontentloaded'

            # Generate the PDF
            await page.pdf(
                path=str(pdf_path_obj),  # page.pdf expects a string path
                format="A4",  # Common paper size (e.g., 'Letter', 'A3')
                print_background=True,  # Include CSS background colors/images
                # Add other options as needed: https://playwright.dev/python/docs/api/class-page#page-pdf
            )

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Ensure the browser is closed even if errors occur
            await browser.close()
