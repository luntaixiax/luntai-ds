from __future__ import annotations
from datetime import date
from typing import List
import pyarrow as pa
import pyarrow.parquet as pq
import ibis
from luntaiDs.CommonTools.SnapStructure.structure import SnapshotDataManagerFileSystem
from luntaiDs.ProviderTools.ssh.sftp import SFTPFileSystem


class SnapshotDataManagerSFTP(SnapshotDataManagerFileSystem):
    """files are saved as parquet snapshots under each schema.table
        file naming convention: dir/tablename_YYYY-MM-DD.parquet
        using fsspec implementation of SFTP system
    """
    @classmethod
    def setup(cls, fs: SFTPFileSystem, root_dir: str):        
        super(SnapshotDataManagerSFTP, cls).setup(
            fs = fs,
            root_dir = root_dir
        )
        
    def read(self, snap_dt: date, **kws) -> ibis.expr.types.Table:
        """Read as ibis dataframe (one snapshot date) data

        :param snap_dt: snap_dt to load
        :return:
        """
        filepath = self.get_file_path(snap_dt=snap_dt)
        
        with self._fs.open(filepath) as obj:
            df_arr: pa.Table = pq.read_table(obj, **kws)
        return self._ibis_con.read_in_memory(df_arr)
        
    
    def reads(self, snap_dts: List[date], **kws) -> ibis.expr.types.Table:
        """reads as ibis dataframe (vertically concat of several given snap dates data)

        :param snap_dts: list of snap dates to read
        :return:
        """
        filepaths = (
            self.get_file_path(snap_dt) for snap_dt in snap_dts
        )
        def df_iter():
            for filepath in filepaths:
                with self._fs.open(filepath) as obj:
                    df_arr: pa.Table = pq.read_table(obj, **kws)
                    yield df_arr
        
        df_arr = pa.concat_tables(df_iter())
        
        return self._ibis_con.read_in_memory(df_arr)
        