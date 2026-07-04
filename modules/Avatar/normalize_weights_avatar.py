"""Optional, user-triggered vertex-weight cleanup for selected mesh(es).

The exporter (export_weights_avatar.py) already guarantees every exported
vertex carries SOME bone weight - it has to, the game's GPU skinning shader
unconditionally reads bone-weight bytes, so there's no such thing as a
truly-unweighted vertex in the file format. When a vertex has no real
weight, the exporter silently borrows its nearest weighted neighbour's
weights (or, if the WHOLE mesh is unweighted, rigid-binds it 100% to one
stable bone) - see `_remap_weights_by_position` / `_rigid_bind_foreign_into_
palette`. That's a safety net, not a visible editing tool: the user only
finds out what happened by reading the export log.

This operator runs the SAME idea on the Blender side, before export, so the
result is visible and adjustable in Weight Paint mode rather than a black
box. It is opt-in only (a button the user clicks), never run automatically.
"""
import bpy
import mathutils

from .export_weights_avatar import _get_armature


class XBG_OT_NormalizeWeights(bpy.types.Operator):
    """Give every vertex at least one bone weight, and make per-vertex
    weights sum to 1.0, without changing which bones currently influence a
    vertex (only their proportions)."""
    bl_idname = "xbg.normalize_weights"
    bl_label = "Normalize Weights to Bones"
    bl_description = (
        "OPTIONAL cleanup, not applied automatically. The game reads "
        "0-255 bone-weight bytes that must sum to ~255 per vertex, and "
        "every vertex needs at least one bone - the exporter already "
        "enforces this silently at export time (nearest-neighbour borrow, "
        "or rigid-bind to one bone if a whole mesh is unweighted). "
        "Running it here lets you see and adjust the result in Weight "
        "Paint mode BEFORE exporting, instead of finding out from the "
        "export log"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        mesh_objects = [o for o in ctx.selected_objects if o.type == 'MESH']
        if not mesh_objects:
            obj = ctx.active_object
            if obj and obj.type == 'MESH':
                mesh_objects = [obj]
        if not mesh_objects:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}

        total_normalized = 0
        total_filled = 0
        total_rigid = 0
        touched = 0

        for obj in mesh_objects:
            arm = _get_armature(obj)
            if not arm:
                continue
            bone_names = {b.name for b in arm.data.bones}
            deform_idx = {vg.index for vg in obj.vertex_groups
                          if vg.name in bone_names}
            # NOTE: deform_idx may legitimately be empty (no vertex groups
            # at all, e.g. every group was deleted) - that's still a mesh
            # we can act on, via the rigid-bind branch below. Only skip
            # objects that have no armature at all (checked above).
            touched += 1

            verts = obj.data.vertices
            weighted_ids = []
            unweighted_ids = []
            for v in verts:
                infl = [ge for ge in v.groups
                        if ge.group in deform_idx and ge.weight > 0.001]
                if infl:
                    weighted_ids.append(v.index)
                else:
                    unweighted_ids.append(v.index)

            if not weighted_ids:
                # Whole mesh unweighted - mirror the exporter's rigid-bind
                # fallback: pin everything to ONE stable deform bone instead
                # of leaving it for the exporter to decide invisibly.
                #
                # CRITICAL: the chosen bone must be a real DEFORM bone the
                # engine actually skins (one present in MB2O / used by the
                # host's other submeshes) - a control bone like 'Root' has
                # NO inverse-bind matrix in MB2O, so the engine would skin
                # this part with a garbage matrix -> invisible/exploded
                # in-game while looking fine in Blender (see agents.md
                # "RIGID-BIND for fully-foreign slices"). The object's own
                # xbg_bone_palette (its original MB2O-indexed palette from
                # import) is the only source we can trust for that -
                # arm.data.bones[0] is NOT safe (often the model/root node).
                bones = arm.data.bones
                palette = obj.get('xbg_bone_palette')
                palette_names = []
                if palette:
                    for gid in palette:
                        if gid is not None and 0 <= gid < len(bones):
                            palette_names.append(bones[gid].name)
                rb_name = obj.get('xbg_rigid_bone')
                pref = ['Pelvis', 'pelvis', 'Spine', 'Spine1', 'Hips',
                        'Bip01_Pelvis', 'Bip01']
                cand_names = ([rb_name] if rb_name else []) + palette_names + pref
                bone_name_set = {b.name for b in bones}
                chosen = next((c for c in cand_names if c in bone_name_set), None)
                if chosen is None and bones:
                    chosen = bones[0].name  # last resort, may not be MB2O-valid
                if chosen is not None:
                    vg = obj.vertex_groups.get(chosen) or obj.vertex_groups.new(name=chosen)
                    vg.add(range(len(verts)), 1.0, 'REPLACE')
                    total_rigid += len(verts)
                    self.report({'WARNING'},
                        f"'{obj.name}' had NO bone weights at all - rigid-bound "
                        f"100% to '{chosen}' (set obj['xbg_rigid_bone'] to "
                        f"pick a different bone)")
                continue

            # KD-tree over weighted-vertex positions, for the borrow step.
            kd = mathutils.kdtree.KDTree(len(weighted_ids))
            for i, vidx in enumerate(weighted_ids):
                kd.insert(verts[vidx].co, vidx)
            kd.balance()

            for vidx in unweighted_ids:
                _, donor_idx, _ = kd.find(verts[vidx].co)
                donor = verts[donor_idx]
                for ge in donor.groups:
                    if ge.group in deform_idx and ge.weight > 0.001:
                        obj.vertex_groups[ge.group].add(
                            [vidx], ge.weight, 'REPLACE')
                total_filled += 1

            # Normalize every (now-)weighted vertex's deform-group weights
            # to sum to 1.0, proportionally - matches what the exporter's
            # 0-255 byte rounding already assumes (total = sum(w) or 1.0).
            for v in verts:
                infl = [ge for ge in v.groups
                        if ge.group in deform_idx and ge.weight > 0.001]
                if not infl:
                    continue
                total = sum(ge.weight for ge in infl)
                if total <= 0:
                    continue
                if abs(total - 1.0) < 1e-4:
                    continue
                for ge in infl:
                    obj.vertex_groups[ge.group].add(
                        [v.index], ge.weight / total, 'REPLACE')
                total_normalized += 1

        if touched == 0:
            self.report({'WARNING'},
                "None of the selected meshes have an armature with bone "
                "vertex groups - nothing to normalize")
            return {'CANCELLED'}

        msg = (f"Normalized {total_normalized} vertex weight(s), "
               f"filled {total_filled} previously-unweighted vertex(es)")
        if total_rigid:
            msg += f", rigid-bound {total_rigid} vertex(es) on fully-unweighted mesh(es)"
        self.report({'INFO'}, msg)
        return {'FINISHED'}
