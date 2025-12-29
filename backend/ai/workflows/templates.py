"""
Workflow Template System for JARVISv3
Enables composition of complex workflows from reusable, validated templates.
"""
import asyncio
from typing import Dict, Any, List, Optional, Callable, Awaitable
from pydantic import BaseModel, Field
from datetime import datetime, UTC
import logging

from .engine import WorkflowEngine, WorkflowNode, NodeType
from ..context.schemas import TaskContext

logger = logging.getLogger(__name__)


class WorkflowTemplate(BaseModel):
    """A reusable workflow template that can be composed into larger workflows"""

    template_id: str
    name: str
    description: str
    version: str = "1.0.0"

    # Template metadata
    category: str  # "research", "code_review", "analysis", "communication"
    complexity: str  # "simple", "medium", "complex"
    estimated_duration: float  # seconds

    # Template definition
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)

    # Input/output specifications
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)

    # Validation and requirements
    required_capabilities: List[str] = Field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = Field(default_factory=list)

    # Template metadata
    author: str = "system"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_validated: Optional[datetime] = None

    def validate_template(self) -> List[str]:
        """Validate template structure and requirements"""
        errors = []

        # Check for required fields
        if not self.nodes:
            errors.append("Template must have at least one node")

        # Check node dependencies
        node_ids = {node["id"] for node in self.nodes}
        for node in self.nodes:
            if "dependencies" in node:
                for dep in node["dependencies"]:
                    if dep not in node_ids:
                        errors.append(f"Node {node['id']} depends on unknown node {dep}")

        # Check edge validity
        for edge in self.edges:
            if edge["from"] not in node_ids:
                errors.append(f"Edge from unknown node {edge['from']}")
            if edge["to"] not in node_ids:
                errors.append(f"Edge to unknown node {edge['to']}")

        return errors


class WorkflowComposer:
    """Composes complex workflows from templates"""

    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.engine = WorkflowEngine()
        logger.info("Workflow Composer initialized")

    def register_template(self, template: WorkflowTemplate) -> bool:
        """Register a workflow template"""
        errors = template.validate_template()
        if errors:
            logger.error(f"Template {template.template_id} validation failed: {errors}")
            return False

        self.templates[template.template_id] = template
        logger.info(f"Registered workflow template: {template.template_id}")
        return True

    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)

    def list_templates(self, category: Optional[str] = None) -> List[WorkflowTemplate]:
        """List available templates, optionally filtered by category"""
        templates = list(self.templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return sorted(templates, key=lambda t: t.name)

    async def compose_workflow(
        self,
        composition_spec: Dict[str, Any],
        context: TaskContext
    ) -> Optional[WorkflowEngine]:
        """
        Compose a workflow from templates

        composition_spec format:
        {
            "name": "composed_workflow_name",
            "templates": [
                {
                    "template_id": "research_template",
                    "instance_id": "research_1",
                    "parameters": {...},
                    "connections": {...}
                }
            ],
            "connections": [
                {"from": "template1.output", "to": "template2.input"}
            ]
        }
        """
        try:
            workflow_name = composition_spec.get("name", "composed_workflow")
            template_specs = composition_spec.get("templates", [])

            # Create new workflow engine for composition
            composed_engine = WorkflowEngine()

            # Track template instances and their nodes
            template_instances = {}

            # Instantiate each template
            for spec in template_specs:
                template_id = spec["template_id"]
                instance_id = spec["instance_id"]
                parameters = spec.get("parameters", {})

                template = self.get_template(template_id)
                if not template:
                    logger.error(f"Template {template_id} not found")
                    return None

                # Instantiate template nodes with prefixed IDs
                instance_nodes = []
                for node_data in template.nodes:
                    # Create node with instance prefix
                    node_id = f"{instance_id}_{node_data['id']}"
                    node = WorkflowNode(
                        id=node_id,
                        type=node_data["type"],
                        description=node_data.get("description", ""),
                        conditions=node_data.get("conditions", {})
                    )

                    # Apply parameters to node conditions
                    if node.conditions:
                        for key, value in parameters.items():
                            if f"{{{key}}}" in str(node.conditions):
                                # Simple parameter substitution
                                node.conditions = self._substitute_parameters(node.conditions, parameters)

                    composed_engine.add_node(node)
                    instance_nodes.append(node_id)

                template_instances[instance_id] = {
                    "template": template,
                    "nodes": instance_nodes,
                    "parameters": parameters
                }

            # Add inter-template connections
            connections = composition_spec.get("connections", [])
            for conn in connections:
                from_spec = conn["from"]  # e.g., "research_1.output"
                to_spec = conn["to"]      # e.g., "analysis_1.input"

                # Parse instance.node format
                from_parts = from_spec.split(".")
                to_parts = to_spec.split(".")

                from_instance = from_parts[0]
                to_instance = to_parts[0]

                if from_instance not in template_instances or to_instance not in template_instances:
                    logger.error(f"Invalid connection: {from_spec} -> {to_spec}")
                    continue

                # Create dependency edge
                from_node = f"{from_instance}_{from_parts[1]}"
                to_node = f"{to_instance}_{to_parts[1]}"

                composed_engine.add_edge(from_node, to_node)

            # Validate composed workflow
            if not self._validate_composition(composed_engine, template_instances):
                logger.error("Workflow composition validation failed")
                return None

            logger.info(f"Successfully composed workflow: {workflow_name}")
            return composed_engine

        except Exception as e:
            logger.error(f"Workflow composition failed: {e}")
            return None

    def _substitute_parameters(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Substitute parameters in template data"""
        if isinstance(data, str):
            for key, value in parameters.items():
                data = data.replace(f"{{{key}}}", str(value))
        elif isinstance(data, dict):
            return {k: self._substitute_parameters(v, parameters) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_parameters(item, parameters) for item in data]
        return data

    def _validate_composition(self, engine: WorkflowEngine, template_instances: Dict[str, Any]) -> bool:
        """Validate that composed workflow is structurally sound"""
        try:
            # Check for orphaned nodes
            all_nodes = set(engine.nodes.keys())
            connected_nodes = set()

            for node in engine.nodes.values():
                connected_nodes.add(node.id)
                connected_nodes.update(node.dependencies)

            orphaned = all_nodes - connected_nodes
            if orphaned:
                logger.warning(f"Composed workflow has orphaned nodes: {orphaned}")

            # Check for cycles using DFS
            visited = set()
            recursion_stack = set()

            def has_cycle(node_id: str) -> bool:
                visited.add(node_id)
                recursion_stack.add(node_id)

                node = engine.nodes.get(node_id)
                if node:
                    for dep in node.dependencies:
                        if dep not in visited:
                            if has_cycle(dep):
                                return True
                        elif dep in recursion_stack:
                            return True

                recursion_stack.remove(node_id)
                return False

            # Check all nodes for cycles
            for node_id in engine.nodes.keys():
                if node_id not in visited:
                    if has_cycle(node_id):
                        logger.error("Composed workflow contains cycles")
                        return False

            return True

        except Exception as e:
            logger.error(f"Composition validation error: {e}")
            return False


# Global composer instance
workflow_composer = WorkflowComposer()


# Template Library - Core validated templates
def initialize_template_library():
    """Initialize the core template library"""

    # Research Template
    research_template = WorkflowTemplate(
        template_id="research_template",
        name="Research Workflow",
        description="Comprehensive research workflow with search and synthesis",
        category="research",
        complexity="medium",
        estimated_duration=30.0,
        nodes=[
            {
                "id": "search",
                "type": NodeType.SEARCH_WEB,
                "description": "Search for information on the topic",
                "conditions": {"query": "{query}"}
            },
            {
                "id": "store_findings",
                "type": NodeType.ACTIVE_MEMORY,
                "description": "Store research findings",
                "dependencies": ["search"],
                "conditions": {"operation": "store", "content": "Research findings for {query}"}
            },
            {
                "id": "synthesize",
                "type": NodeType.LLM_WORKER,
                "description": "Synthesize findings into coherent summary",
                "dependencies": ["store_findings"]
            }
        ],
        input_schema={"query": "string"},
        output_schema={"summary": "string", "sources": "array"},
        required_capabilities=["web_search", "memory", "llm"]
    )

    # Code Review Template
    code_review_template = WorkflowTemplate(
        template_id="code_review_template",
        name="Code Review Workflow",
        description="Automated code review with analysis and recommendations",
        category="code_review",
        complexity="medium",
        estimated_duration=45.0,
        nodes=[
            {
                "id": "analyze_code",
                "type": NodeType.LLM_WORKER,
                "description": "Analyze code for issues and patterns",
                "conditions": {"code": "{code_content}"}
            },
            {
                "id": "check_security",
                "type": NodeType.VALIDATOR,
                "description": "Security validation of code",
                "dependencies": ["analyze_code"]
            },
            {
                "id": "generate_feedback",
                "type": NodeType.LLM_WORKER,
                "description": "Generate review feedback and recommendations",
                "dependencies": ["check_security"]
            }
        ],
        input_schema={"code_content": "string", "language": "string"},
        output_schema={"issues": "array", "recommendations": "array", "score": "number"},
        required_capabilities=["llm", "security_validation"]
    )

    # Analysis Template
    analysis_template = WorkflowTemplate(
        template_id="analysis_template",
        name="Data Analysis Workflow",
        description="Analyze data or content with insights and recommendations",
        category="analysis",
        complexity="complex",
        estimated_duration=60.0,
        nodes=[
            {
                "id": "extract_insights",
                "type": NodeType.LLM_WORKER,
                "description": "Extract key insights from data",
                "conditions": {"data": "{input_data}"}
            },
            {
                "id": "validate_insights",
                "type": NodeType.VALIDATOR,
                "description": "Validate extracted insights",
                "dependencies": ["extract_insights"]
            },
            {
                "id": "generate_recommendations",
                "type": NodeType.LLM_WORKER,
                "description": "Generate actionable recommendations",
                "dependencies": ["validate_insights"]
            },
            {
                "id": "store_analysis",
                "type": NodeType.ACTIVE_MEMORY,
                "description": "Store analysis results",
                "dependencies": ["generate_recommendations"],
                "conditions": {"operation": "store", "content": "Analysis results for {topic}"}
            }
        ],
        input_schema={"input_data": "string", "topic": "string"},
        output_schema={"insights": "array", "recommendations": "array", "confidence": "number"},
        required_capabilities=["llm", "memory", "validation"]
    )

    # Register templates
    templates = [research_template, code_review_template, analysis_template]
    for template in templates:
        if not workflow_composer.register_template(template):
            logger.error(f"Failed to register template: {template.template_id}")

    logger.info(f"Initialized template library with {len(templates)} templates")


# Initialize on import
initialize_template_library()
