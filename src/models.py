from pydantic import BaseModel, Field
import uuid
from typing import List


class MinimalSource(BaseModel):
    """A single source location: a file path and a character range."""
    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(BaseModel):
    """A chunk of a source file with its character range and text."""
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    heading: str = ""


class UnansweredQuestion(BaseModel):
    """A question without an answer, identified by a generated id."""

    question_id: str = Field(default_factory=lambda:
                             str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth sources and answer."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A collection of RAG questions, answered or not."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for one question."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources and the generated answer for one question."""
    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a set of questions, with k."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results with generated answers, with k."""
    search_results: List[MinimalAnswer]
    k: int
