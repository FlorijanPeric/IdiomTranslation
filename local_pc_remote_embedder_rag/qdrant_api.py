import json
from urllib import request
from urllib.error import HTTPError, URLError


class QdrantHttpClient:
    def __init__(self, url: str, api_key: str | None = None, timeout: int = 30):
        self.url = url.rstrip("/")
        self.api_key = api_key or ""
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        return headers

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = request.Request(
            url=f"{self.url}{path}",
            method=method,
            data=data,
            headers=self._headers(),
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
            if not raw:
                return {}
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Qdrant HTTP error ({error.code}): {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Qdrant connection failed: {error}") from error

    def list_collections(self) -> list[str]:
        parsed = self._request("GET", "/collections")
        result = parsed.get("result", {})
        items = result.get("collections", []) if isinstance(result, dict) else []
        names: list[str] = []
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.append(item["name"])
        return names

    def create_collection(self, collection_name: str, vector_size: int) -> None:
        payload = {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine",
            }
        }
        self._request("PUT", f"/collections/{collection_name}", payload)

    def delete_collection(self, collection_name: str) -> None:
        self._request("DELETE", f"/collections/{collection_name}")

    def upsert_points(self, collection_name: str, points: list[dict], wait: bool = True) -> None:
        payload = {"points": points}
        suffix = "?wait=true" if wait else ""
        self._request("PUT", f"/collections/{collection_name}/points{suffix}", payload)

    def count_points(self, collection_name: str) -> int:
        parsed = self._request("POST", f"/collections/{collection_name}/points/count", {"exact": False})
        result = parsed.get("result", {})
        if isinstance(result, dict):
            return int(result.get("count", 0))
        return 0

    def search(self, collection_name: str, query_vector: list[float], limit: int = 5) -> list[dict]:
        payload = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
        }
        parsed = self._request("POST", f"/collections/{collection_name}/points/search", payload)
        result = parsed.get("result", [])
        if isinstance(result, list):
            return result
        return []
