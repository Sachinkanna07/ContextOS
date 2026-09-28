"""Deterministic offline connector used by tests and benchmarks."""
from __future__ import annotations
from contextos.connectors.models import ConnectorItem
class FakeConnector:
    source_type="fake"
    def __init__(self,connector_id:str,items:list[ConnectorItem])->None:self.connector_id,self.items=connector_id,items;self.failure:Exception|None=None
    async def health(self)->bool:return self.failure is None
    async def close(self)->None:pass
    async def scan(self,cursor:str|None)->tuple[list[ConnectorItem],str|None]:
        if self.failure: raise self.failure
        return list(self.items),str(len(self.items))
