from pathlib import Path
import io
import fitz
from PIL import Image
from docling.document_converter import DocumentConverter

class PDFParser:

    def __init__(self,processed_dir="data/processed",images_dir="data/extracted_images"):
          self.converter=DocumentConverter()
          self.processed_dir=Path(processed_dir)
          self.images_dir=Path(images_dir)
          self._create_output_folders()
          
    def _create_output_folders(self):
         self.processed_dir.mkdir(parents=True, exist_ok=True)
         self.images_dir.mkdir(parents=True, exist_ok=True) 
        
##reading and parsing the pdf using docling converter which gives document object as result 
##that has map of text title tables and hierarchies !!
    def parse_pdf(self,pdf_path):
         pdf_path=Path(pdf_path)
         result=self.converter.convert(pdf_path)
         document=result.document
         return document
##using docling we use format unformatted text strings using docling to turn into strucutred document
##into markdown prserves headings,bold_text and table cleanly
    def save_markdown(self,document,output_name):
         markdown=document.export_to_markdown()##this will export documetn to markdown format in markdwon varible
         output_path=self.processed_dir/f"{output_name}.md"

         with open(output_path,"w",encoding="utf-8") as f: ##writing the markdown to file in processed dir
            f.write(markdown)
        
            return output_path


##this function switches to library fitz to find images embedded inside pdf pages
##loops through every single page and find images on taht page
    def extract_images(self, pdf_path):
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)
        image_paths = []
        
        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            image_list = page.get_images(full=True)
            
            for image_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image = Image.open(io.BytesIO(image_bytes))
                image_name = f"page_{page_number+1}_image_{image_index+1}.{image_ext}"
                
                save_path = self.images_dir / image_name
                image.save(save_path)
                image_paths.append(save_path)

        # ✅ Moved out of the loops so it processes the entire document first!
        return image_paths


## orchestration of the entire process instead of calling all 3 function us this

    def get_document(self,pdf_path):
        pdf_path=Path(pdf_path)
        document=self.parse_pdf(pdf_path)
        markdown_path=self.save_markdown(
            document,
            pdf_path.stem
        )
        image_paths=self.extract_images(pdf_path)

        return{
            "document":document,
            "markdown":markdown_path,
            "image_paths":image_paths
        }
