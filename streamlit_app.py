import streamlit as st
from chains import build_rag_chain
from embeddings import build_faiss_index, load_faiss_index
from loaders import load_documents
from dspy_pipeline import IEPPipeline, LessonPlanPipeline
import os
import tempfile
import shutil
import time
import json
from datetime import datetime
from langchain.schema import Document
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import zipfile

# Page Configuration
st.set_page_config(
    page_title="Educational Assistant",
    page_icon=":books:",
    layout="wide"
)

# Initialize directories
DATA_DIR = "data"
INDEX_DIR = "models/faiss_index"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Initialize session state
if "chain" not in st.session_state:
    st.session_state["chain"] = None
if "documents_processed" not in st.session_state:
    st.session_state["documents_processed"] = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "iep_results" not in st.session_state:
    st.session_state["iep_results"] = []
if "documents" not in st.session_state:
    st.session_state["documents"] = []
if "lesson_plans" not in st.session_state:
    st.session_state["lesson_plans"] = []

def process_uploaded_file(uploaded_file, use_dspy=False):
    """Process an uploaded file and update the RAG chain."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save uploaded file
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Debug print
        print(f"Processing file: {uploaded_file.name}")
        
        # Load documents
        documents = load_documents(temp_dir)
        if not documents:
            st.error("No text could be extracted from the uploaded file.")
            return False
        
        # Store documents in session state
        st.session_state["documents"].extend(documents)
        
        # Build index
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        os.makedirs(INDEX_DIR)
        
        vectorstore = build_faiss_index(documents, INDEX_DIR)
        
        if vectorstore:
            st.session_state["chain"] = build_rag_chain(vectorstore)
            return True
        return False

def create_lesson_plan_pdf(plan_data):
    """Create a formatted PDF from lesson plan data."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph(f"Lesson Plan - {plan_data['timeframe'].title()}", title_style))
    story.append(Spacer(1, 12))

    # Sections
    for section, content in plan_data.items():
        if section not in ['timeframe', 'timestamp', 'source_iep', 'quality_score']:
            # Section header
            story.append(Paragraph(section.replace('_', ' ').title(), styles['Heading2']))
            story.append(Spacer(1, 6))
            
            # Section content
            if isinstance(content, list):
                for item in content:
                    story.append(Paragraph(f"• {item}", styles['Normal']))
            else:
                story.append(Paragraph(str(content), styles['Normal']))
            story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)
    return buffer

st.title("Educational Assistant with GPT-4o Mini")

# Sidebar for API Key and File Upload
with st.sidebar:
    st.title("Document Upload")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    if not api_key:
        st.error("Please provide your OpenAI API key.")

    use_dspy = st.checkbox("Use DSPy Processing", value=False)

    uploaded_files = st.file_uploader(
        "Upload educational documents",
        type=["txt", "docx", "pdf", "md"],
        accept_multiple_files=True
    )

    if uploaded_files and not st.session_state["documents_processed"]:
        processing_success = True  # Track overall success
        with st.spinner("Processing documents..."):
            st.write("### Processing Files")
            for file in uploaded_files:
                status_container = st.empty()
                status_container.info(f"Processing {file.name}...")
                
                try:
                    success = process_uploaded_file(file, use_dspy=use_dspy)
                    if success:
                        status_container.success(f"Successfully processed {file.name}")
                    else:
                        status_container.error(f"Failed to process {file.name}")
                except Exception as e:
                    status_container.error(f"Error processing {file.name}: {str(e)}")
            
            if processing_success:
                st.session_state["documents_processed"] = True
                st.success("All documents processed successfully!")
            else:
                st.error("Error processing some documents.")

    if st.session_state["documents_processed"]:
        if st.button("Clear Documents"):
            st.session_state["documents_processed"] = False
            st.session_state["chain"] = None
            st.session_state["documents"] = []  # Clear documents
            st.session_state["iep_results"] = []  # Clear IEP results
            if os.path.exists(INDEX_DIR):
                shutil.rmtree(INDEX_DIR)
            st.experimental_rerun()

    st.title("System Status")
    if st.button("Check System Health"):
        status = check_system_health()
        
        st.write("### System Components Status")
        for component, is_healthy in status.items():
            if is_healthy:
                st.success(f"✅ {component}: OK")
            else:
                st.error(f"❌ {component}: Failed")
                
        # Show detailed status
        with st.expander("System Details"):
            st.write("- API Key:", "Present" if status["api_key"] else "Missing")
            st.write("- Directories:", "Created" if status["directories"] else "Failed")
            st.write("- Document Loading:", "Working" if status["document_loading"] else "Failed")
            st.write("- Vector Store:", "Operational" if status["vectorstore"] else "Failed")
            st.write("- RAG Chain:", "Functional" if status["chain"] else "Failed")
            st.write("- DSPy Integration:", "Available" if status["dspy"] else "Not Available")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Document Q&A", "Chat", "IEP Generation", "Lesson Plans"])

# Document Q&A Tab
with tab1:
    st.header("Document Q&A")

    query = st.text_area(
        "Ask a question about the documents:",
        placeholder="Example: Can you summarize the main points?",
        disabled=not st.session_state["documents_processed"]
    )

    if query:
        if st.session_state["chain"]:
            with st.spinner("Generating response..."):
                try:
                    start_time = time.time()
                    
                    # Get response
                    chain_response = st.session_state["chain"]({"query": query})
                    end_time = time.time()
                    
                    # Extract answer and sources
                    answer = chain_response.get('result', '')
                    sources = chain_response.get('source_documents', [])
                    
                    # Display response
                    st.write("### Response")
                    st.write(answer)
                    
                    # Display sources
                    with st.expander("View Retrieved Context"):
                        if sources:
                            for i, doc in enumerate(sources, 1):
                                st.write(f"Source {i}:")
                                st.write(doc.page_content)
                                st.write("---")
                        else:
                            st.write("No source documents were retrieved.")
                            # Debug info
                            st.write("Debug Info:")
                            st.write(f"Chain type: {type(st.session_state['chain'])}")
                            st.write(f"Response type: {type(chain_response)}")
                            st.write(f"Response keys: {chain_response.keys() if isinstance(chain_response, dict) else 'Not a dict'}")
                    
                    st.write(f"Response time: {end_time - start_time:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.error("Full error:", exc_info=True)
        else:
            st.warning("Please upload documents first!")

# Chat Interface Tab
with tab2:
    st.header("Chat Interface")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        if not st.session_state.get("documents_processed"):
            st.error("Please upload documents first!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Use the call method instead of run
                        chain_response = st.session_state["chain"]({"query": prompt})
                        response = chain_response.get('result', '')
                        sources = chain_response.get('source_documents', [])
                        
                        # Add response to messages
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.markdown(response)
                        
                        # Show sources in expander
                        with st.expander("View Source Context"):
                            if sources:
                                for i, doc in enumerate(sources, 1):
                                    st.write(f"Source {i}:")
                                    st.write(doc.page_content)
                                    st.write("---")
                            else:
                                st.write("No source documents were retrieved.")
                                
                    except Exception as e:
                        error_message = f"Error generating response: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        st.error(error_message)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

# IEP Generation Tab
with tab3:
    st.header("IEP Generation")
    
    if st.session_state["documents_processed"]:
        # Add debug information
        st.write(f"Number of documents loaded: {len(st.session_state.get('documents', []))}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("IEP Generation Controls")
            generate_button = st.button(
                "Generate IEPs",
                help="Click to generate IEPs from uploaded documents"
            )
            
            if generate_button:
                try:
                    with st.spinner("Generating IEPs... This may take a few minutes."):
                        # Get the pipeline
                        pipeline = IEPPipeline()
                        
                        # Process documents and store results
                        iep_results = []
                        for doc in st.session_state.get("documents", []):
                            # Add debug print
                            st.write(f"Processing document: {doc.metadata.get('source', 'Unknown')}")
                            
                            result = pipeline.process_documents([doc])
                            if result:
                                iep_data = {
                                    "source": doc.metadata.get("source", "Unknown"),
                                    "timestamp": datetime.now().isoformat(),
                                    "content": result[0].page_content,
                                    "metadata": result[0].metadata
                                }
                                iep_results.append(iep_data)
                                # Add debug print
                                st.write(f"Successfully processed: {iep_data['source']}")
                        
                        st.session_state["iep_results"] = iep_results
                        st.success(f"Successfully generated {len(iep_results)} IEPs!")
                
                except Exception as e:
                    st.error(f"Error generating IEPs: {str(e)}")
                    st.error("Full error:", exc_info=True)
        
        with col2:
            if st.session_state["iep_results"]:
                st.subheader("Bulk Download")
                # Create a ZIP file containing all IEPs
                import io
                import zipfile
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for idx, iep in enumerate(st.session_state["iep_results"]):
                        json_data = json.dumps(iep, indent=2)
                        zip_file.writestr(f"IEP_{idx + 1}.json", json_data)
                
                st.download_button(
                    label="Download All IEPs (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="all_ieps.zip",
                    mime="application/zip"
                )
        
        # Display IEP results
        if st.session_state["iep_results"]:
            st.subheader("Generated IEPs")
            
            for idx, iep in enumerate(st.session_state["iep_results"]):
                with st.expander(f"IEP {idx + 1} - {iep['source']}", expanded=False):
                    # Display formatted content
                    st.markdown("### Content")
                    st.markdown(iep['content'])
                    
                    # Display metadata
                    st.markdown("### Metadata")
                    st.json(iep['metadata'])
                    
                    # Individual download button
                    json_data = json.dumps(iep, indent=2)
                    st.download_button(
                        label=f"Download IEP {idx + 1}",
                        data=json_data,
                        file_name=f"IEP_{idx + 1}.json",
                        mime="application/json",
                        key=f"download_iep_{idx}"  # Unique key for each button
                    )
        
        # Clear results button
        if st.session_state["iep_results"]:
            if st.button("Clear IEP Results"):
                st.session_state["iep_results"] = []
                st.experimental_rerun()
                
    else:
        st.warning("Please upload and process documents first!")
        st.info("Once documents are processed, you can generate IEPs here.")

# Lesson Plan Generation Tab
with tab4:
    st.header("Lesson Plan Generation")
    
    # Combined form for all lesson plan generation
    st.subheader("Lesson Plan Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("lesson_plan_form"):
            # Required form fields
            st.markdown("### Basic Information")
            subject = st.text_input("Subject *", placeholder="e.g., Mathematics, Reading, Science")
            grade_level = st.text_input("Grade Level *", placeholder="e.g., 3rd Grade, High School")
            
            # Timeframe selection
            timeframe = st.radio(
                "Schedule Type *",
                ["Daily", "Weekly"],
                help="Choose between a daily lesson plan or a weekly schedule"
            )
            
            duration = st.text_input(
                "Daily Duration *", 
                placeholder="e.g., 45 minutes per session"
            )
            
            if timeframe == "Weekly":
                days_per_week = st.multiselect(
                    "Select Days *",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                )
            
            st.markdown("### Learning Details")
            specific_goals = st.text_area(
                "Specific Learning Goals *",
                placeholder="Enter specific goals for this lesson, one per line"
            )
            
            materials = st.text_area(
                "Materials Needed",
                placeholder="List required materials, one per line"
            )
            
            st.markdown("### Additional Support")
            additional_accommodations = st.text_area(
                "Additional Accommodations",
                placeholder="Enter any specific accommodations beyond those in the IEP"
            )
            
            # IEP Selection
            st.markdown("### IEP Integration")
            if st.session_state.get("iep_results"):
                selected_iep = st.selectbox(
                    "Select IEP to Integrate *",
                    options=[iep["source"] for iep in st.session_state["iep_results"]],
                    format_func=lambda x: f"IEP from {x}"
                )
            else:
                st.error("No IEPs available. Please generate an IEP first.")
                selected_iep = None
            
            st.markdown("*Required fields")
            
            generate_button = st.form_submit_button("Generate Enhanced Lesson Plan")
            
            if generate_button:
                # Validate required fields
                if not selected_iep:
                    st.error("Please generate an IEP first before creating a lesson plan.")
                elif not all([subject, grade_level, specific_goals, duration]):
                    st.error("Please fill in all required fields (marked with *).")
                elif timeframe == "Weekly" and not days_per_week:
                    st.error("Please select at least one day for weekly schedule.")
                else:
                    try:
                        with st.spinner(f"Generating {timeframe} Lesson Plan..."):
                            # Get IEP data
                            iep_data = next(
                                iep for iep in st.session_state["iep_results"] 
                                if iep["source"] == selected_iep
                            )
                            
                            # Combine form and IEP data
                            combined_data = {
                                "iep_content": iep_data.get("content", ""),
                                "subject": subject,
                                "grade_level": grade_level,
                                "duration": duration,
                                "days": days_per_week if timeframe == "Weekly" else None,
                                "specific_goals": specific_goals.split('\n'),
                                "materials": materials.split('\n') if materials else [],
                                "additional_accommodations": additional_accommodations.split('\n') if additional_accommodations else [],
                                "timeframe": timeframe.lower(),
                                "source_iep": selected_iep
                            }
                            
                            # Generate enhanced plan
                            pipeline = LessonPlanPipeline()
                            plan = pipeline.generate_lesson_plan(combined_data, timeframe.lower())
                            
                            if plan:
                                st.session_state["lesson_plans"].append(plan)
                                st.success(f"Enhanced {timeframe.lower()} lesson plan generated successfully!")
                            
                    except Exception as e:
                        st.error(f"Error generating lesson plan: {str(e)}")

    # Update the display section to handle daily/weekly plans
    if st.session_state.get("lesson_plans"):
        st.markdown("### Generated Lesson Plans")
        
        for idx, plan in enumerate(st.session_state["lesson_plans"]):
            with st.expander(f"{plan['timeframe'].title()} Plan - {plan['subject']}", expanded=True):
                # Basic info
                st.markdown(f"**Grade Level**: {plan['grade_level']}")
                st.markdown(f"**Duration**: {plan['duration']}")
                
                # Schedule
                st.markdown("### Schedule")
                if isinstance(plan['schedule'], dict):  # Weekly schedule
                    for day, activities in plan['schedule'].items():
                        st.markdown(f"#### {day}")
                        st.markdown(activities)
                else:  # Daily schedule
                    st.markdown(plan['schedule'])
                
                # Learning objectives and strategies
                st.markdown("### Learning Objectives")
                for obj in plan['learning_objectives']:
                    st.markdown(f"- {obj}")
                
                st.markdown("### Instructional Strategies")
                for strategy in plan['instructional_strategies']:
                    st.markdown(f"- {strategy}")
                
                # Assessment and modifications
                st.markdown("### Assessment Criteria")
                for criterion in plan['assessment_criteria']:
                    st.markdown(f"- {criterion}")
                
                st.markdown("### Modifications & Accommodations")
                for mod in plan['modifications']:
                    st.markdown(f"- {mod}")
                
                # Download button
                pdf_buffer = create_lesson_plan_pdf(plan)
                st.download_button(
                    label=f"Download {plan['timeframe'].title()} Plan (PDF)",
                    data=pdf_buffer.getvalue(),
                    file_name=f"lesson_plan_{plan['subject']}_{plan['timeframe']}_{idx + 1}.pdf",
                    mime="application/pdf"
                )

# Footer
st.markdown("---")
st.markdown("Educational Assistant powered by GPT-4o Mini and LangChain")
