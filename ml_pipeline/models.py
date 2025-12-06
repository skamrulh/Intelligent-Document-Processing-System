import random
from typing import Dict, List, Any

class ComplianceNERModel:
    """
    Extracting entities relevant to compliance (Dates, Orgs, Regulations).
    In production, this would be a HuggingFace BERT model fine-tuned on legal text.
    
    """
    def extract_entities(self, filename: str) -> Dict[str, Any]:
        # Simulating extraction based on synthetic logic for the demo
        return {
            "transformer_entities": [
                {"entity": "ORG", "word": "Acme Corp", "score": 0.98},
                {"entity": "REGULATION", "word": "GDPR", "score": 0.95},
                {"entity": "DATE", "word": "2024-01-01", "score": 0.99}
            ],
            "doc_type": "contract" if "contract" in filename.lower() else "report"
        }

class RiskClassifier:
    """
    Classifies the risk level of a document.
    """
    def classify_risk(self, filename: str, entities: List[Dict]) -> Dict[str, Any]:
        # Simple heuristic for demo purposes:
        # If 'violation' or 'breach' is inferred (randomized for demo variety), flaging high risk
        base_risk = random.choice(["LOW", "MEDIUM", "HIGH"])
        
        return {
            "risk_level": base_risk,
            "confidence": 0.89,
            "flags": ["Missing audit trail"] if base_risk == "HIGH" else []
        }

class SemanticSearchEngine:
    """
    Handles document indexing and retrieval.
    In production, this would use Pinecone or Milvus and Sentence Transformers.

    """
    def __init__(self):
        self.index = []

    def add_document(self, doc_id: str, metadata: Dict):
        self.index.append({"id": doc_id, **metadata})

    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        # Here mocking semantic search return
        return [
            {
                "id": doc["id"],
                "score": random.uniform(0.7, 0.99),
                "snippet": f"Context matching '{query}' found in document..."
            }
            for doc in self.index[:k]
        ]