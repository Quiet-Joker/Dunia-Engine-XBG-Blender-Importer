import struct,os,tempfile,bpy
from bpy_extras.image_utils import load_image
from ..Core.debug import VerboseLogger

class XBTConverter:
    _temp_files = {}
    _temp_cleanup_list = []  # Files to clean up at the end
    
    @staticmethod
    def get_temp_texture_path(xbt_path, import_as_dds=False):
        """Get temporary texture file path. If import_as_dds is False, convert to PNG using Blender."""
        cache_key = f"{xbt_path}_{import_as_dds}"
        if cache_key in XBTConverter._temp_files and os.path.exists(XBTConverter._temp_files[cache_key]):
            return XBTConverter._temp_files[cache_key]
        
        # First extract DDS data from XBT
        dds_data = XBTConverter.convert_to_dds(xbt_path)
        if not dds_data:
            return None
        
        try:
            basename = os.path.splitext(os.path.basename(xbt_path))[0]
            hash_val = hash(xbt_path) & 0xFFFFFFFF
            
            if import_as_dds:
                # Legacy mode: save as DDS
                VerboseLogger.log(f"⚠ WARNING: Importing {basename} as DDS - texture painting will be corrupted!")
                temp_path = os.path.join(tempfile.gettempdir(), f"xbg_import_{basename}_{hash_val}.dds")
                with open(temp_path, 'wb') as f:
                    f.write(dds_data)
                XBTConverter._temp_files[cache_key] = temp_path
                return temp_path
            else:
                # New default mode: convert DDS to PNG using Blender
                temp_dds = os.path.join(tempfile.gettempdir(), f"xbg_temp_{basename}_{hash_val}.dds")
                temp_png = os.path.join(tempfile.gettempdir(), f"xbg_import_{basename}_{hash_val}.png")
                
                # Write DDS file
                with open(temp_dds, 'wb') as f:
                    f.write(dds_data)
                
                # Add to cleanup list
                XBTConverter._temp_cleanup_list.append(temp_dds)
                
                try:
                    # Load DDS using Blender's image utility (handles DDS formats properly)
                    img = load_image(temp_dds, check_existing=False)
                    
                    if not img or img.size[0] == 0 or img.size[1] == 0:
                        if img:
                            bpy.data.images.remove(img)
                        raise Exception("Image failed to load or has invalid dimensions")
                    
                    VerboseLogger.log(f"Converting {basename} to PNG ({img.size[0]}x{img.size[1]}, {img.channels} channels)")
                    
                    # Save as PNG using the simple method
                    img.filepath_raw = temp_png
                    img.file_format = 'PNG'
                    img.save()
                    
                    # Clean up the image from Blender's memory
                    bpy.data.images.remove(img)
                    
                    # Verify PNG was created and is valid
                    if not os.path.exists(temp_png):
                        raise Exception("PNG file was not created")
                    
                    if os.path.getsize(temp_png) < 100:
                        try:
                            os.remove(temp_png)
                        except:
                            pass
                        raise Exception("PNG file is too small (likely corrupt)")
                    
                    VerboseLogger.log(f"✓ Successfully converted {basename} to PNG")
                    
                    # Success! Return PNG path
                    XBTConverter._temp_files[cache_key] = temp_png
                    return temp_png
                    
                except Exception as e:
                    # PNG conversion failed - use DDS as fallback
                    VerboseLogger.log(f"⚠ PNG conversion failed for {basename}: {e}")
                    VerboseLogger.log(f"  Using DDS format for this texture")
                    
                    # Clean up failed PNG if it exists
                    if os.path.exists(temp_png):
                        try:
                            os.remove(temp_png)
                        except:
                            pass
                    
                    # Return the DDS file path
                    XBTConverter._temp_files[cache_key] = temp_dds
                    return temp_dds
                    
        except Exception as e:
            VerboseLogger.log(f"Error processing texture {xbt_path}: {e}")
            return None
    
    @staticmethod
    def get_temp_dds_path(xbt_path):
        """Convenience alias — force the DDS import path.

        Equivalent to `get_temp_texture_path(xbt_path, import_as_dds=True)`.
        The UI toggle 'Import XBT as DDS' is what users actually flip;
        this alias is kept for external callers / scripts that want the
        DDS branch without having to pass the flag.
        """
        return XBTConverter.get_temp_texture_path(xbt_path, import_as_dds=True)

    @staticmethod
    def cleanup_temp_files():
        """Clean up all temporary files created during import"""
        VerboseLogger.log("\nCleaning up temporary texture files...")
        cleaned = 0
        
        # Clean up main cached files
        for path in XBTConverter._temp_files.values():
            try:
                if os.path.exists(path):
                    os.remove(path)
                    cleaned += 1
            except Exception as e:
                VerboseLogger.log(f"Warning: Could not delete {os.path.basename(path)}: {e}")
        
        # Clean up temp DDS files from cleanup list
        for path in XBTConverter._temp_cleanup_list:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    cleaned += 1
            except Exception as e:
                VerboseLogger.log(f"Warning: Could not delete temp file {os.path.basename(path)}: {e}")
        
        XBTConverter._temp_files.clear()
        XBTConverter._temp_cleanup_list.clear()
        
        if cleaned > 0:
            VerboseLogger.log(f"✓ Cleaned up {cleaned} temporary file(s)")
        
        # Final sweep: clean up any orphaned temp files
        try:
            temp_dir = tempfile.gettempdir()
            orphaned = 0
            for filename in os.listdir(temp_dir):
                if filename.startswith('xbg_temp_') or filename.startswith('xbg_import_'):
                    filepath = os.path.join(temp_dir, filename)
                    try:
                        if os.path.isfile(filepath):
                            os.remove(filepath)
                            orphaned += 1
                    except:
                        pass
            if orphaned > 0:
                VerboseLogger.log(f"✓ Cleaned up {orphaned} orphaned temp file(s)")
        except:
            pass
    
    @staticmethod
    def convert_to_dds(xbt_path):
        try:
            with open(xbt_path, 'rb') as f:
                xbt_data = f.read()
            if xbt_data[:3] == b'TBX':
                if len(xbt_data) >= 12:
                    header_size = struct.unpack('<I', xbt_data[8:12])[0]
                    dds_data = xbt_data[header_size:] if 32 <= header_size <= 1024 and header_size < len(xbt_data) else xbt_data[32:]
                else:
                    dds_data = xbt_data[32:]
            else:
                dds_data = xbt_data
            if len(dds_data) >= 4 and dds_data[:4] == b'DDS ':
                return dds_data
            for offset in [64, 128, 256]:
                if len(xbt_data) > offset:
                    test_data = xbt_data[offset:]
                    if len(test_data) >= 4 and test_data[:4] == b'DDS ':
                        return test_data
            return None
        except:
            return None
    
    @staticmethod
    def find_mip0_variant(texture_path, data_folder):
        if '_mip0.xbt' in texture_path.lower():
            return texture_path
        mip0_path = texture_path.replace('.xbt', '_mip0.xbt')
        return mip0_path if os.path.exists(os.path.join(data_folder, mip0_path.replace('\\', os.sep).replace('/', os.sep))) else None


def read_dds(xbt_bytes):
    """Inverse of build_xbt (matches XBTConverter.convert_to_dds)."""
    if xbt_bytes[:3] == b'TBX':
        hs = struct.unpack_from('<I', xbt_bytes, 8)[0]
        if 32 <= hs <= 1024 and hs < len(xbt_bytes):
            d = xbt_bytes[hs:]
        else:
            d = xbt_bytes[32:]
    else:
        d = xbt_bytes
    return d if d[:4] == b'DDS ' else None
