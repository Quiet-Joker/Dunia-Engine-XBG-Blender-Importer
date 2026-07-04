import xml.etree.ElementTree as ET
import os
import mathutils

try:
    from ..Core.debug import VerboseLogger as vlog
except:
    class vlog:
        @staticmethod
        def log(m): pass
        @staticmethod
        def log_bone(*a): pass
        @staticmethod
        def log_bone_world_transform(*a): pass
        @staticmethod
        def log_xml_bone(*a): pass


class Bone:
    __slots__ = ('name', 'parent_id', 'local_rotation_quat', 'local_position', 'local_matrix', 'world_matrix', 'bind_matrix', 'mb2o_index')
    
    def __init__(self):
        self.name = None
        self.parent_id = None
        self.local_rotation_quat = None
        self.local_position = [0, 0, 0]
        self.local_matrix = None
        self.world_matrix = None
        self.bind_matrix = None  # MB2O inverse bind matrix
        self.mb2o_index = None  # Index into MB2O array


class Skeleton:
    def __init__(self):
        self.bones = []
        self.bone_to_mb2o_map = {}  # Maps bone_id → MB2O index
    
    def add_bone(self, bone):
        self.bones.append(bone)
    
    def get_bone_count(self):
        return len(self.bones)
    
    def compute_bone_transforms(self):
        vlog.log("\n=== COMPUTING BONE TRANSFORMS ===")
        for i, bone in enumerate(self.bones):
            if bone.local_rotation_quat is None:
                continue
            
            rot_matrix = bone.local_rotation_quat.to_matrix().to_4x4()
            pos_matrix = mathutils.Matrix.Translation(bone.local_position)
            bone.local_matrix = pos_matrix @ rot_matrix
            
            if bone.parent_id is not None and 0 <= bone.parent_id < len(self.bones):
                parent = self.bones[bone.parent_id]
                bone.world_matrix = parent.world_matrix @ bone.local_matrix if parent.world_matrix is not None else bone.local_matrix
            else:
                bone.world_matrix = bone.local_matrix
            
            if bone.world_matrix:
                vlog.log_bone_world_transform(i, bone.name, tuple(bone.world_matrix.translation))
    
    def _mb2o_root_frame(self):
        """The outermost root bone's FULL world transform (translation +
        rotation). MB2O inverse-bind matrices are expressed relative to this
        frame, not world space. See modules/Avatar/skeleton_avatar.py for
        the full writeup - this is the same fix, ported to FC2 (the code
        here was a byte-identical copy of the same bug).
        """
        for i, bd in enumerate(self.bones):
            pid = bd.parent_id
            if (pid is None or pid < 0 or pid == i) and bd.world_matrix:
                return bd.world_matrix.copy()
        return mathutils.Matrix.Identity(4)

    def apply_bind_matrices(self, bind_matrices, sub_mesh_list=None):
        """Match MB2O inverse-bind matrices to bones by GEOMETRIC cross-check
        against EDON, not by guessing the on-disk array order.

        MB2O is a flat array with no in-file index->bone label. The old
        approach guessed the array order matched "unique bones encountered
        while walking DNKS submesh palettes" — verified WRONG (see
        modules/Avatar/skeleton_avatar.py and agents.md "MB2O import-side
        verification", 2026-06-30): wrong assignments displaced bones by
        1.3-2.3 meters, with whole bones' matrices landing on completely
        different bones.

        EDON's hierarchy walk (compute_bone_transforms) already
        reconstructs every bone's correct bind-pose position independent
        of MB2O, so we use it as ground truth: each on-disk MB2O matrix is
        assigned to whichever EDON bone's own (already-correct) world
        position it lands closest to, once un-inverted into the root
        bone's frame. Matching is a single global greedy pass over ALL
        (mb2o_index, bone) pairs sorted by ascending distance.

        Bones that don't get a confident MB2O match keep their EDON world
        transform — this is the existing/already-correct fallback.
        """
        vlog.log(f"\n=== APPLYING MB2O BIND MATRICES (geometric cross-check vs EDON) ===")
        vlog.log(f"  MB2O matrices available: {len(bind_matrices)}")

        root_frame = self._mb2o_root_frame()
        candidates = {i: b.world_matrix.translation for i, b in enumerate(self.bones)
                      if b.world_matrix is not None}

        DIST_OK = 0.12  # 12cm: real matches cluster <1cm even on packed
                         # face-bone regions; real mismatches are 10cm-2m+.
        pairs = []
        for k, m in enumerate(bind_matrices):
            try:
                mb2o_world = (root_frame @ m.inverted()).translation
            except Exception:
                vlog.log(f"  MB2O[{k}]: matrix not invertible, skipped")
                continue
            for i, p in candidates.items():
                d = (mb2o_world - p).length
                if d <= DIST_OK:
                    pairs.append((d, k, i))
        pairs.sort(key=lambda t: t[0])

        claimed_k, claimed_i = set(), set()
        for d, k, i in pairs:
            if k in claimed_k or i in claimed_i:
                continue
            bone = self.bones[i]
            bone.bind_matrix = bind_matrices[k]
            bone.mb2o_index = k
            self.bone_to_mb2o_map[i] = k
            claimed_k.add(k)
            claimed_i.add(i)
            vlog.log(f"  Bone {i} ({bone.name}): MB2O[{k}] applied (dist={d:.5f})")

        unmatched_mb2o = len(bind_matrices) - len(claimed_k)
        bones_without_mb2o = len(self.bones) - len(claimed_i)
        vlog.log(f"  Matched {len(claimed_k)}/{len(bind_matrices)} MB2O matrices to bones")
        if unmatched_mb2o:
            vlog.log(f"  {unmatched_mb2o} MB2O matrix(es) had no confident bone match "
                      f"(within {DIST_OK}m) and were not applied")
        if bones_without_mb2o:
            vlog.log(f"  {bones_without_mb2o} bones have no MB2O data (using EDON transforms)")


class XMLBoneData:
    def __init__(self, name, position, rotation, parent=None):
        self.name = name
        self.position = position
        self.rotation = rotation
        self.parent = parent


def quaternion_from_xbg_data(qd):
    return mathutils.Quaternion((qd[3], qd[0], qd[1], qd[2])) if len(qd) >= 4 else mathutils.Quaternion()


def parse_skeleton_chunk(g, skeleton):
    w = g.i(3)
    bone_count = w[2]
    vlog.log(f"\n=== EDON CHUNK (Skeleton) ===\nBone Count: {bone_count}")
    
    for m in range(bone_count):
        bone = Bone()
        g.b(4)
        w = g.i(3)
        
        quat_data = g.f(4)
        bone.local_rotation_quat = quaternion_from_xbg_data(quat_data)
        
        pos_data = g.f(3)
        bone.local_position = list(pos_data)
        
        g.f(3)
        g.i(1)
        g.f(1)
        g.i(1)
        
        name_len = g.i(1)[0]
        bone.name = g.word(name_len)[-25:]
        bone.parent_id = w[2]
        g.b(1)
        
        vlog.log_bone(m, bone.name, bone.parent_id, pos_data, quat_data)
        skeleton.add_bone(bone)
    
    skeleton.compute_bone_transforms()


def parse_mb2o_chunk(g):
    """Parse MB2O chunk containing inverse bind matrices
    
    IMPORTANT: MB2O matrices are stored in COLUMN-MAJOR format!
    The matrices must be transposed when reading.
    """
    vlog.log("\n=== MB2O CHUNK (Bind Matrices) ===")
    
    g.i(2)  # Skip first two ints
    matrix_count = g.i(1)[0]
    vlog.log(f"  Reading {matrix_count} MB2O inverse bind matrices")
    
    matrices = []
    for i in range(matrix_count):
        # Read 16 floats for 4x4 matrix
        # CRITICAL: XBG stores matrices in COLUMN-MAJOR format!
        # File layout: [c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, ...]
        #   where c = column, r = row
        matrix_data = g.f(16)
        
        # Convert from column-major to Blender's Matrix format
        # Blender Matrix constructor takes rows, so we transpose by reading columns as rows
        mat = mathutils.Matrix((
            (matrix_data[0], matrix_data[4], matrix_data[8],  matrix_data[12]),  # Row 0 from column data
            (matrix_data[1], matrix_data[5], matrix_data[9],  matrix_data[13]),  # Row 1 from column data
            (matrix_data[2], matrix_data[6], matrix_data[10], matrix_data[14]),  # Row 2 from column data
            (matrix_data[3], matrix_data[7], matrix_data[11], matrix_data[15])   # Row 3 from column data
        ))
        
        matrices.append(mat)
        
        if vlog.enabled and i < 3:  # Only log first 3 for brevity
            trans = mat.translation
            vlog.log(f"  Matrix {i}: Translation = ({trans.x:.3f}, {trans.y:.3f}, {trans.z:.3f})")
    
    vlog.log(f"  Successfully parsed {len(matrices)} MB2O inverse bind matrices")
    vlog.log(f"  Note: These are indexed by DNKS bone palette order, NOT bone ID!")
    return matrices


class XMLSkeletonParser:
    @staticmethod
    def find_xml_file(xbg_filepath):
        base_path = os.path.splitext(xbg_filepath)[0]
        xml_path = base_path + '.xml'
        if os.path.exists(xml_path):
            vlog.log(f"\n{'='*60}\nFound XML file: {xml_path}\n{'='*60}")
            return xml_path
        return None
    
    @staticmethod
    def parse_xml_skeleton(xml_filepath):
        try:
            tree = ET.parse(xml_filepath)
            root = tree.getroot()
            bones = {}
            mesh_to_bone = {}
            mesh_index_to_bone = {}
            mesh_index_to_name = {}
            
            descriptor = root.find('.//descriptor')
            if descriptor is None:
                return bones, mesh_to_bone, mesh_index_to_bone, mesh_index_to_name
            
            graphic_component = descriptor.find(".//component[@class='GraphicComponent']")
            if graphic_component is not None:
                vlog.log("\n=== XML MESH-TO-BONE MAPPINGS ===")
                for obj in graphic_component.findall('object'):
                    mesh_name = obj.get('meshName')
                    bone_name = obj.get('boneName')
                    mesh_index = obj.get('index')
                    
                    if mesh_name and bone_name:
                        mesh_to_bone[mesh_name.upper()] = bone_name
                        vlog.log(f"  {mesh_name} → {bone_name}")
                    
                    if mesh_index is not None and bone_name:
                        try:
                            idx = int(mesh_index)
                            mesh_index_to_bone[idx] = bone_name
                            mesh_index_to_name[idx] = mesh_name if mesh_name else None
                        except:
                            pass
            
            skeleton = graphic_component.find('.//skeleton') if graphic_component else None
            if skeleton is None:
                return bones, mesh_to_bone, mesh_index_to_bone, mesh_index_to_name
            
            vlog.log(f"\n=== XML SKELETON PARSING ===")
            
            def parse_bone(bone_elem, parent_name=None):
                name = bone_elem.get('name')
                if not name:
                    return
                
                pos_str = bone_elem.get('pos', '0,0,0')
                pos = tuple(float(x) for x in pos_str.split(','))
                
                rot_str = bone_elem.get('rot', '1,0,0,0')
                rot = tuple(float(x) for x in rot_str.split(','))
                
                bones[name] = XMLBoneData(name, pos, rot, parent_name)
                vlog.log_xml_bone(name, pos, rot, parent_name)
                
                for child_bone in bone_elem.findall('bone'):
                    parse_bone(child_bone, name)
            
            for bone_elem in skeleton.findall('bone'):
                parse_bone(bone_elem, None)
            
            vlog.log(f"\nTotal XML bones: {len(bones)}")
            return bones, mesh_to_bone, mesh_index_to_bone, mesh_index_to_name
        except:
            return {}, {}, {}, {}
