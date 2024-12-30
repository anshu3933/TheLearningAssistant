import dspy
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import os
from datetime import datetime

class IEPPipeline:
    """Information Extraction and Processing Pipeline using DSPy."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize the DSPy pipeline.
        
        Args:
            model_name (str): Name of the OpenAI model to use
        """
        # Initialize DSPy components
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        self.lm = dspy.OpenAI(model=model_name)
        dspy.settings.configure(lm=self.lm)
        
        # Define extraction signature
        class InformationExtractor(dspy.Signature):
            """Extract structured information from text."""
            context = dspy.InputField(desc="Input text to process")
            insights = dspy.OutputField(desc="Key insights and main points")
            entities = dspy.OutputField(desc="Important entities and concepts")
            summary = dspy.OutputField(desc="Concise summary")
        
        self.extractor = dspy.Predict(InformationExtractor)
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process a list of documents through the DSPy pipeline.
        
        Args:
            documents (List[Document]): Original documents to process
            
        Returns:
            List[Document]: Enhanced documents with DSPy processing results
        """
        enhanced_docs = []
        
        for doc in documents:
            try:
                # Process with DSPy
                result = self.extractor(context=doc.page_content)
                
                # Create enhanced content
                enhanced_content = f"""
Original Content:
{doc.page_content}

Key Insights:
{result.insights}

Important Entities:
{result.entities}

Summary:
{result.summary}
"""
                
                # Create enhanced document
                enhanced_docs.append(Document(
                    page_content=enhanced_content,
                    metadata={
                        **doc.metadata,
                        "processed_with": "dspy",
                        "enhancement_type": "full",
                        "original_length": len(doc.page_content)
                    }
                ))
                
                print(f"Successfully processed document: {doc.metadata.get('source', 'unknown')}")
                
            except Exception as e:
                print(f"Error in DSPy processing: {e}")
                # Fall back to original document
                enhanced_docs.append(doc)
        
        return enhanced_docs

def build_faiss_index_with_dspy(documents: List[Document], 
                               persist_directory: str,
                               model_name: str = "gpt-4o-mini") -> Optional[FAISS]:
    """Build a FAISS index with DSPy-enhanced documents.
    
    Args:
        documents (List[Document]): Original documents to process
        persist_directory (str): Directory to save the FAISS index
        model_name (str): Name of the OpenAI model to use for DSPy
        
    Returns:
        Optional[FAISS]: Enhanced FAISS vectorstore
    """
    try:
        # Initialize DSPy pipeline
        pipeline = IEPPipeline(model_name=model_name)
        
        # Process documents
        enhanced_docs = pipeline.process_documents(documents)
        
        # Combine original and enhanced documents
        all_docs = documents + enhanced_docs
        
        # Build and return index
        from embeddings import build_faiss_index
        return build_faiss_index(all_docs, persist_directory)
        
    except Exception as e:
        print(f"Error in DSPy-enhanced indexing: {e}")
        # Fall back to regular indexing
        from embeddings import build_faiss_index
        return build_faiss_index(documents, persist_directory)

class LessonPlanRM(dspy.Signature):
    """Signature for the reasoning module."""
    context = dspy.InputField()
    reasoning = dspy.OutputField()

class LessonPlanSignature(dspy.Signature):
    """Signature for generating lesson plans with reasoning."""
    
    # Input fields with detailed descriptions
    iep_content = dspy.InputField(desc="Full IEP content including student needs and accommodations")
    subject = dspy.InputField(desc="Subject area (e.g., Math, Science)")
    grade_level = dspy.InputField(desc="Student's grade level")
    duration = dspy.InputField(desc="Length of each lesson")
    specific_goals = dspy.InputField(desc="Specific learning objectives to be achieved")
    materials = dspy.InputField(desc="Required teaching materials and resources")
    additional_accommodations = dspy.InputField(desc="Additional accommodations beyond IEP requirements")
    timeframe = dspy.InputField(desc="Daily or weekly planning timeframe")
    days = dspy.InputField(desc="Days of the week for instruction")
    
    # Output fields with detailed structure requirements
    schedule = dspy.OutputField(desc="""
        Detailed daily schedule including:
        - Warm-up activities (5-10 minutes)
        - Main concept introduction with visual aids
        - Guided practice with accommodations
        - Independent work time
        - Assessment and closure
        Minimum length: 200 words
    """)
    
    lesson_plan = dspy.OutputField(desc="""
        Comprehensive lesson plan including:
        1. Detailed teaching strategies
        2. Step-by-step instructions
        3. Differentiation methods
        4. IEP accommodations integration
        5. Real-world connections
        6. Student engagement techniques
        7. Time management details
        Minimum length: 300 words
    """)
    
    learning_objectives = dspy.OutputField(desc="""
        Specific, measurable objectives including:
        - Knowledge acquisition goals
        - Skill development targets
        - Application objectives
        - Assessment criteria
        Minimum 5 detailed objectives
    """)
    
    assessment_criteria = dspy.OutputField(desc="""
        Detailed assessment criteria including:
        - Understanding checks
        - Skill demonstration requirements
        - Progress monitoring methods
        - Success indicators
        Minimum 5 specific criteria
    """)
    
    modifications = dspy.OutputField(desc="""
        Specific IEP-aligned modifications including:
        - Learning accommodations
        - Assessment modifications
        - Environmental adjustments
        - Support strategies
        Minimum 5 detailed modifications
    """)
    
    instructional_strategies = dspy.OutputField(desc="""
        Detailed teaching strategies including:
        - Visual learning techniques
        - Hands-on activities
        - Technology integration
        - Differentiation methods
        - Student engagement approaches
        Minimum 5 specific strategies
    """)

class LessonPlanPipeline:
    """Pipeline for generating adaptive lesson plans from IEPs."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.lm = dspy.OpenAI(
            model=model_name,
            max_tokens=2000  # Increase token limit
        )
        
        dspy.settings.configure(lm=self.lm)
        
        self.reasoning_module = dspy.ChainOfThought(LessonPlanRM)
        self.generator = dspy.ChainOfThought(LessonPlanSignature)
        
        # Enhanced prompt template
        self.prompt_template = """
        As an experienced special education teacher, create a detailed and comprehensive lesson plan following these steps:
        
        1. Analyze the IEP requirements and student needs:
        - Review all accommodations and modifications needed
        - Identify specific learning style preferences
        - Note particular challenges and strengths
        - Consider past performance and progress
        
        2. Design comprehensive grade-level learning objectives:
        - Align with curriculum standards
        - Break down complex concepts into manageable parts
        - Set both short-term and long-term goals
        - Include measurable outcomes
        
        3. Create detailed accommodations and modifications:
        - Incorporate all IEP requirements
        - Design multiple forms of visual and hands-on activities
        - Plan differentiated instruction for various skill levels
        - Include technology and multi-sensory approaches
        
        4. Develop a structured yet flexible lesson plan:
        - Create detailed timeline with buffer time
        - Include varied activities for different learning styles
        - Plan transition strategies
        - Incorporate regular check-ins and assessments
        
        5. Design assessment and progress monitoring:
        - Include multiple forms of assessment
        - Plan for ongoing progress monitoring
        - Create success criteria
        - Design feedback mechanisms
        
        Ensure all components are detailed, specific, and aligned with student needs.
        Minimum length for each section: 200 words
        """
        
        print("Successfully initialized LessonPlanPipeline")
        
        # Add detailed example for better zero-shot learning
        self.example = {
            "iep_content": "Student requires extended time, visual aids, and hands-on learning",
            "subject": "Mathematics",
            "grade_level": "3rd Grade",
            "duration": "45 minutes",
            "specific_goals": "Understanding Pythagoras theorem through practical applications",
            "materials": "Cardboard triangles, measuring tape, grid paper",
            "additional_accommodations": "Connect to real-world examples, provide visual supports",
            "timeframe": "weekly",
            "days": "Monday through Friday",
            "schedule": """
                Daily Schedule Pattern:
                1. Warm-up (5 min): Review previous concepts using visual aids
                2. Introduction (10 min): Present new concept with real-world examples
                3. Guided Practice (15 min): Hands-on activities with manipulatives
                4. Independent Work (10 min): Practice with support as needed
                5. Closure (5 min): Quick assessment and preview next day
            """,
            "lesson_plan": """
                Week-long Progression:
                Monday: Introduction to right triangles using real objects
                Tuesday: Exploring square numbers with grid paper
                Wednesday: Discovering Pythagoras pattern with manipulatives
                Thursday: Applying theorem to real-world problems
                Friday: Review and creative applications
                
                Teaching Methodology:
                - Use visual aids consistently
                - Incorporate hands-on activities
                - Connect to real-world examples
                - Provide frequent checks for understanding
                - Allow extended time as needed
            """,
            "learning_objectives": [
                "Identify right triangles in real-world objects",
                "Calculate missing sides using Pythagoras theorem",
                "Apply theorem to solve practical problems",
                "Demonstrate understanding through multiple methods"
            ],
            "assessment_criteria": [
                "Accurate identification of right triangles",
                "Correct calculation of missing sides",
                "Proper use of theorem in applications",
                "Clear explanation of process"
            ],
            "modifications": [
                "Extended time for calculations",
                "Use of calculator when needed",
                "Visual step-by-step guides",
                "Reduced problem set with increased depth"
            ],
            "instructional_strategies": [
                "Multi-sensory approach to learning",
                "Regular comprehension checks",
                "Peer learning opportunities",
                "Visual and hands-on demonstrations"
            ]
        }
    
    def _format_list_to_string(self, items: List[str]) -> str:
        """Convert a list of items to a numbered string."""
        if not items:
            return ""
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
    
    def generate_lesson_plan(self, data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """Generate a detailed daily or weekly lesson plan."""
        try:
            print("Starting lesson plan generation...")
            
            # Format list inputs to strings
            specific_goals_str = "\n".join(data["specific_goals"]) if isinstance(data["specific_goals"], list) else str(data["specific_goals"])
            materials_str = "\n".join(data.get("materials", [])) if isinstance(data.get("materials"), list) else str(data.get("materials", ""))
            accommodations_str = "\n".join(data.get("additional_accommodations", [])) if isinstance(data.get("additional_accommodations"), list) else str(data.get("additional_accommodations", ""))
            days_str = ", ".join(data.get("days", [])) if isinstance(data.get("days"), list) else str(data.get("days", ""))
            
            # First, generate reasoning about the plan
            context = f"""
            IEP Content: {data['iep_content']}
            Subject: {data['subject']}
            Grade Level: {data['grade_level']}
            Duration: {data['duration']}
            Goals: {specific_goals_str}
            
            Instructions: {self.prompt_template}
            """
            
            reasoning = self.reasoning_module(context=context).reasoning
            
            # Then generate the actual plan using the reasoning
            result = self.generator(
                iep_content=data["iep_content"],
                subject=data["subject"],
                grade_level=data["grade_level"],
                duration=data["duration"],
                specific_goals=specific_goals_str,
                materials=materials_str,
                additional_accommodations=accommodations_str,
                timeframe=timeframe,
                days=days_str
            )
            
            # Process and format the result
            plan_data = {
                "timestamp": datetime.now().isoformat(),
                "timeframe": timeframe,
                "subject": data["subject"],
                "grade_level": data["grade_level"],
                "duration": data["duration"],
                "reasoning": reasoning,
                "schedule": result.schedule,
                "lesson_plan": result.lesson_plan,
                "learning_objectives": self._process_field(result, 'learning_objectives'),
                "assessment_criteria": self._process_field(result, 'assessment_criteria'),
                "modifications": self._process_field(result, 'modifications'),
                "instructional_strategies": self._process_field(result, 'instructional_strategies'),
                "source_iep": data["source_iep"],
                "quality_score": self.evaluate_lesson_plan(result)
            }
            
            return plan_data
            
        except Exception as e:
            print(f"Detailed error in generate_lesson_plan: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return self._generate_basic_plan(data, timeframe)
    
    def _generate_basic_plan(self, data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """Generate a basic lesson plan when the main generation fails."""
        try:
            # Create a simplified plan structure
            basic_plan = {
                "timestamp": datetime.now().isoformat(),
                "timeframe": timeframe,
                "subject": data["subject"],
                "grade_level": data["grade_level"],
                "duration": data["duration"],
                "schedule": self._create_basic_schedule(data, timeframe),
                "lesson_plan": "Basic lesson plan structure",
                "learning_objectives": data["specific_goals"],
                "assessment_criteria": ["Basic understanding of concepts", "Completion of exercises"],
                "modifications": data.get("additional_accommodations", []),
                "instructional_strategies": ["Visual aids", "Hands-on activities"],
                "source_iep": data["source_iep"],
                "quality_score": 0.5  # Basic quality score
            }
            return basic_plan
        except Exception as e:
            print(f"Error in basic plan generation: {str(e)}")
            return None
    
    def _create_basic_schedule(self, data: Dict[str, Any], timeframe: str) -> str:
        """Create a basic schedule structure."""
        if timeframe.lower() == "weekly":
            days = data.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
            return "\n".join([f"{day}: {data['duration']} - {data['subject']} lesson" for day in days])
        else:
            return f"Single day lesson - {data['duration']}"    
    def _process_field(self, result, field_name: str) -> List[str]:
        """Process a result field, handling both string and list formats."""
        value = getattr(result, field_name, [])
        if isinstance(value, str):
            return [item.strip() for item in value.split('\n') if item.strip()]
        elif isinstance(value, list):
            return value
        return []
    
    def evaluate_lesson_plan(self, plan: Dict[str, Any]) -> float:
        """Evaluate the quality of a generated lesson plan."""
        try:
            score = 0.0
            
            # Check segmentation
            if isinstance(plan.get('schedule'), str) and len(plan.get('schedule', '').split('\n')) > 2:
                score += 0.3
                
            # Check real-world anchoring
            lesson_plan_text = str(plan.get('lesson_plan', ''))
            if 'real-world' in lesson_plan_text.lower() or 'application' in lesson_plan_text.lower():
                score += 0.3
                
            # Check IEP alignment
            if plan.get('modifications') and len(plan.get('modifications', [])) > 0:
                score += 0.4
                
            return score
            
        except Exception as e:
            print(f"Error in evaluate_lesson_plan: {str(e)}")
            return 0.0

def process_iep_to_lesson_plans(documents: List[Document], 
                              timeframes: List[str] = ["initial", "mid-year", "annual"]) -> List[Document]:
    """Process IEPs and generate lesson plans for multiple timeframes."""
    pipeline = LessonPlanPipeline()
    enhanced_docs = []
    
    for doc in documents:
        try:
            # First, generate IEP data
            iep_pipeline = IEPPipeline()
            iep_result = iep_pipeline.process_documents([doc])[0]
            
            # Then generate lesson plans for each timeframe
            for timeframe in timeframes:
                lesson_plan = pipeline.generate_lesson_plan(
                    iep_data={"content": iep_result.page_content, "source": doc.metadata.get("source")},
                    timeframe=timeframe
                )
                
                if lesson_plan:
                    # Evaluate the plan
                    quality_score = pipeline.evaluate_lesson_plan(lesson_plan)
                    lesson_plan["quality_score"] = quality_score
                    
                    # Create a new document
                    enhanced_docs.append(Document(
                        page_content=str(lesson_plan),
                        metadata={
                            **doc.metadata,
                            "type": "lesson_plan",
                            "timeframe": timeframe,
                            "quality_score": quality_score,
                            "source_iep": doc.metadata.get("source")
                        }
                    ))
                    
        except Exception as e:
            print(f"Error processing document for lesson plans: {e}")
            
    return enhanced_docs
