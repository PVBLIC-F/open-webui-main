from fpdf import FPDF
from open_webui.models.chats import ChatTitleMessagesForm


class PDFGenerator:
    """Simple PDF generator for chat messages."""

    def __init__(self, form_data: ChatTitleMessagesForm):
        self.form_data = form_data

    def generate_chat_pdf(self) -> bytes:
        """Generate a simple, efficient PDF from chat messages."""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            pdf.set_auto_page_break(auto=True, margin=15)

            # Add title
            pdf.set_font_size(14)
            pdf.cell(0, 10, self.form_data.title, ln=True, align='C')
            pdf.ln(5)

            # Convert messages to simple text format
            for message in self.form_data.messages:
                role = message.get("role", "user").upper()
                content = message.get("content", "")
                
                # Simple text processing - let FPDF handle the formatting
                text = f"### {role}\n{content}\n\n"
                
                # Use FPDF's built-in text handling
                pdf.set_font_size(10)
                for line in text.split('\n'):
                    if pdf.get_y() > 270:  # Near page bottom
                        pdf.add_page()
                    pdf.cell(0, 5, line, ln=True)

            return bytes(pdf.output())
            
        except Exception as e:
            # Simple fallback
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, f"Chat: {self.form_data.title}", ln=True)
            return bytes(pdf.output())
