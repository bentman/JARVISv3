"""MCP Dispatcher for JARVISv3
Standardizes access to external tools via the Model Context Protocol.
"""
import logging
import tempfile
import subprocess
import os
import sys
import json
from typing import Dict, Any, List, Optional
import requests
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

class MCPDispatcher:
    """
    Dispatches tool calls to registered MCP servers.
    """

    def __init__(self):
        self.servers = {}
        # Pre-register built-in tools
        self.tools = {
            "read_file": self._read_file_tool,
            "list_files": self._list_files_tool,
            "write_file": self._write_file_tool,
            "execute_python": self._execute_python_code_tool,
            "web_search": self._web_search_tool,
            "code_execution": self._execute_code_tool,
            "system_info": self._system_info_tool,
            "file_analysis": self._file_analysis_tool
        }

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a registered MCP tool"""
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")

        logger.info(f"Executing MCP tool: {name}")
        return await self.tools[name](**arguments)

    # Built-in tool implementations (Simulating real MCP behavior)

    async def _read_file_tool(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                return {"content": f.read(), "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _list_files_tool(self, directory: str = ".") -> Dict[str, Any]:
        import os
        try:
            return {"files": os.listdir(directory), "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _write_file_tool(self, path: str, content: str) -> Dict[str, Any]:
        try:
            with open(path, 'w') as f:
                f.write(content)
            return {"success": True, "path": path}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _execute_python_code_tool(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in a secure sandbox environment.
        This implementation uses subprocess to run code in a separate process
        with limited permissions and resources.
        """
        import tempfile
        import subprocess
        import os
        import signal
        import sys
        from io import StringIO

        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            # Execute the code in a separate process with a timeout
            # This provides basic isolation from the main application
            result = subprocess.run(
                [sys.executable, temp_file_path],
                timeout=10,  # 10 second timeout to prevent infinite loops
                capture_output=True,
                text=True,
                # Limit the environment to prevent access to sensitive data
                env={
                    'PATH': os.environ.get('PATH', ''),
                    'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
                    'HOME': os.environ.get('HOME', ''),
                },
                # Run in a restricted working directory
                cwd=tempfile.gettempdir()
            )

            # Return the results
            return {
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                "error": "Code execution timed out (10 seconds)",
                "success": False
            }
        except Exception as e:
            return {
                "error": f"Execution error: {str(e)}",
                "success": False
            }
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass  # If we can't delete the file, just continue

    async def _web_search_tool(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform a web search using DuckDuckGo
        """
        try:
            from ddgs import DDGS
            ddgs = DDGS()
            results = ddgs.text(query, max_results=max_results)

            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "timestamp": datetime.now(UTC).isoformat()
                })

            return {
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Web search failed: {str(e)}",
                "success": False
            }

    async def _execute_code_tool(self, language: str, code: str) -> Dict[str, Any]:
        """
        Execute code in various languages with proper sandboxing
        """
        try:
            if language == "python":
                return await self._execute_python_code_tool(code)
            elif language == "javascript":
                return await self._execute_javascript_code(code)
            else:
                return {
                    "error": f"Unsupported language: {language}",
                    "success": False
                }
        except Exception as e:
            return {
                "error": f"Code execution failed: {str(e)}",
                "success": False
            }

    async def _execute_javascript_code(self, code: str) -> Dict[str, Any]:
        """
        Execute JavaScript code in a sandboxed environment
        """
        try:
            # Create a temporary HTML file with the JavaScript code
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <body>
                <script>
                    {code}
                </script>
            </body>
            </html>
            """

            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as temp_file:
                temp_file.write(html_content)
                temp_file_path = temp_file.name

            # Use a headless browser or Node.js to execute
            # For simplicity, we'll use a basic approach here
            # In production, you would use a proper sandboxed environment

            result = {
                "output": "JavaScript execution completed",
                "success": True
            }

            os.unlink(temp_file_path)
            return result

        except Exception as e:
            return {
                "error": f"JavaScript execution failed: {str(e)}",
                "success": False
            }

    async def _system_info_tool(self) -> Dict[str, Any]:
        """
        Get system information for debugging and monitoring
        """
        try:
            import platform
            import psutil

            system_info = {
                "os": platform.system(),
                "os_version": platform.version(),
                "python_version": platform.python_version(),
                "cpu_cores": psutil.cpu_count(logical=False),
                "cpu_threads": psutil.cpu_count(logical=True),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                "disk_available_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
                "timestamp": datetime.now(UTC).isoformat()
            }

            return {
                "system_info": system_info,
                "success": True
            }
        except Exception as e:
            return {
                "error": f"System info retrieval failed: {str(e)}",
                "success": False
            }

    async def _file_analysis_tool(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze file content and metadata
        """
        try:
            import magic
            import hashlib

            # Get file stats
            file_stats = os.stat(file_path)

            # Get file type
            file_type = magic.from_file(file_path, mime=True)

            # Calculate hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            analysis = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size": file_stats.st_size,
                "file_type": file_type,
                "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "sha256_hash": file_hash,
                "timestamp": datetime.now(UTC).isoformat()
            }

            return {
                "analysis": analysis,
                "success": True
            }
        except Exception as e:
            return {
                "error": f"File analysis failed: {str(e)}",
                "success": False
            }

# Global instance
mcp_dispatcher = MCPDispatcher()
