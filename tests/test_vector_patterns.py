import copy
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from drawcv import (Path, Point, Rectangle, Circle, Group, VectorPattern, FillStyle,
                    StrokeStyle, Color, Transform, Scene, OpenCVRenderer,
                    LinearGradient, GradientStop, ValidationError, RenderError)
from drawcv.core.paint_sampling import sample_paint
from drawcv.styles.paint import paint_from_dict
from drawcv.effects.clipping import ClipRect


def tile(color=Color(255,0,0)):
    return Rectangle(width=8, height=8, fill=FillStyle(color=color)).to_path()


def render(node):
    scene = Scene(160,120,background=Color(0,0,0,0)); scene.add(node)
    return OpenCVRenderer().render(scene,alpha=True).buffer


def test_spacing_origin_and_transparency_are_periodic():
    paint = VectorPattern(tile(), 8, 8, spacing=(8,8), origin=Point(3,5), opacity=.5)
    image = render(Rectangle(width=160,height=120,fill=FillStyle(paint=paint)))
    for x,y in [(6,8),(22,8),(6,24),(150,104)]:
        np.testing.assert_allclose(image[y,x], [0,0,255,128], atol=1)
    for x,y in [(14,8),(6,16),(30,16)]:
        assert image[y,x,3] == 0


def test_full_tile_has_no_seams_and_alpha_is_not_multiplied_at_edges():
    paint = VectorPattern(tile(), 8, 8, opacity=.5)
    image = render(Rectangle(width=160,height=120,fill=FillStyle(paint=paint)))
    assert image[2:-2,2:-2,3].min() >= 127
    assert image[2:-2,2:-2,3].max() <= 128


@pytest.mark.parametrize('space', ['object','world'])
@pytest.mark.parametrize('stroke', [False, True])
def test_native_svg_patterns_match_raster_interior(space, stroke):
    resvg = pytest.importorskip('resvg_py')
    art = Group(children=[tile(), Circle(center=Point(13,13),radius=3,fill=FillStyle(color=Color(0,0,255))).to_path()])
    paint = VectorPattern(art, 18, 18, spacing=(3,2), origin=Point(2,3), space=space,
                          transform=Transform(rotation=12, pivot=Point(0,0)))
    node = Rectangle(position=Point(10,10),width=85,height=70,
                fill=None if stroke else FillStyle(paint=paint),
                stroke=StrokeStyle(width=14, paint=paint,space='object') if stroke else None,
                transform=Transform.from_matrix(np.array([[1.3,.2,10],[.1,1.1,4],[0,0,1.]])))
    node.clip = ClipRect(0,0,100,90) if stroke else ClipRect(15,15,70,55)
    scene=Scene(160,120,background=Color(0,0,0,0));scene.add(node)
    before = copy.deepcopy(paint.to_dict())
    actual = OpenCVRenderer().render(scene,alpha=True).buffer
    exported = scene.export_svg(strict=True)
    root = ET.fromstring(exported.svg)
    assert root.find('.//{http://www.w3.org/2000/svg}pattern') is not None
    assert root.find('.//{http://www.w3.org/2000/svg}image') is None
    expected = cv2.imdecode(np.frombuffer(resvg.svg_to_bytes(svg_string=exported.svg),np.uint8),-1)
    # At the clip boundary, a tiny motif can be entirely absent in one rasterizer.
    # Compare pattern coverage inside the clip, independently of that boundary.
    from drawcv.effects.clipping import evaluate_clip_coverage
    kernel=np.ones((5,5),np.uint8)
    clip_inside=cv2.erode((evaluate_clip_coverage(node.clip,node,160,120)>.99).astype(np.uint8),kernel)>0
    solid = node.clone()
    if stroke:
        solid.stroke.paint = Color(255,255,255)
    else:
        solid.fill.paint = Color(255,255,255)
    host_inside = cv2.erode((render(solid)[...,3]==255).astype(np.uint8),kernel)>0
    interior=cv2.erode((expected[...,3]==255).astype(np.uint8),kernel)>0
    exterior=cv2.dilate((expected[...,3]>0).astype(np.uint8),kernel)==0
    assert interior.any() and actual[...,3][interior].min()>240
    assert not actual[...,3][exterior & clip_inside & host_inside].any()
    assert paint.to_dict()==before


def test_json_copy_live_edits_and_old_schema():
    art=tile()
    paint=VectorPattern(art,16,16)
    clone=paint.copy()
    assert clone.to_dict()==paint.to_dict() and clone.artwork is not art
    assert paint_from_dict(paint.to_dict()).to_dict()==paint.to_dict()
    node=Rectangle(width=100,height=100,fill=FillStyle(paint=paint))
    original=render(node)
    art.fill.color=Color(0,255,0)
    assert not np.array_equal(original,render(node))
    scene=Scene(160,120);scene.add(node)
    restored=Scene.from_json(scene.to_json()).objects[0]
    assert restored.fill.paint.to_dict()==paint.to_dict()
    from drawcv import CURRENT_SCHEMA_VERSION, SchemaMigrator
    assert CURRENT_SCHEMA_VERSION=='1.12'
    old=Scene(10,10).to_dict();old['version']='1.11'
    assert Scene.from_dict(SchemaMigrator.migrate(old)).width==10


def test_tile_edge_clips_overflow_without_changing_source_parent():
    art=Rectangle(position=Point(-5,-5),width=30,height=30,fill=FillStyle(color=Color(255,0,0))).to_path()
    parent=Group(children=[art],transform=Transform(translation_x=20))
    paint=VectorPattern(art,8,8,spacing=(8,8))
    before=art.to_dict()
    image=render(Rectangle(width=160,height=120,fill=FillStyle(paint=paint)))
    assert image[4,4,3]==255 and image[12,12,3]==0
    assert art.parent is parent and art.to_dict()==before


@pytest.mark.parametrize('kwargs',[{'width':0},{'height':float('inf')},{'opacity':True},
    {'spacing':(-1,0)},{'space':'screen'},{'origin':(0,0)}])
def test_validation(kwargs):
    options=dict(width=8,height=8);options.update(kwargs)
    with pytest.raises(ValidationError):
        VectorPattern(tile(),**options)


def test_nested_patterns_and_invalid_mutation_fail_cleanly():
    art=tile();pattern=VectorPattern(art,8,8)
    with pytest.raises(ValidationError):
        pattern.width=-1
    assert pattern.width==8
    art.fill=FillStyle(paint=pattern)
    with pytest.raises(ValidationError,match='Nested'):
        pattern.to_dict()
    with pytest.raises(ValidationError,match='Nested'):
        sample_paint(pattern,np.eye(3),10,10)


def test_singular_pattern_is_transparent_and_large_tile_has_clear_limit():
    pattern=VectorPattern(tile(),8,8,transform=Transform.from_matrix(np.diag([0.,1.,1.])))
    assert not sample_paint(pattern,np.eye(3),10,10).any()
    pattern.transform=Transform(scale_x=10000)
    with pytest.raises(RenderError,match='working limit'):
        sample_paint(pattern,np.eye(3),10,10)


def test_group_opacity_and_shared_world_gradient_in_tile():
    gradient=LinearGradient(Point(0,0),Point(16,0),
        (GradientStop(0,Color(255,0,0)),GradientStop(1,Color(0,0,255))),space='world')
    left=tile();right=tile()
    left.fill=FillStyle(paint=gradient);right.fill=FillStyle(paint=gradient)
    right.move(8,0)
    pattern=VectorPattern(Group(children=[left,right],opacity=.5),16,8)
    normalized=pattern.tile().children[0].children
    # Shared world paints are converted independently for each artwork transform.
    assert normalized[0].fill.paint is not normalized[1].fill.paint
    assert gradient.space=='world'
    image=render(Rectangle(width=160,height=120,fill=FillStyle(paint=pattern)))
    assert image[3,3,2] > image[3,3,0]
    assert image[3,12,0] > image[3,12,2]
    np.testing.assert_allclose(image[3,[3,7,8,12],3],128,atol=1)


def test_bgr_alpha_compositing_and_paint_transform_edit():
    pattern=VectorPattern(tile(),8,8,opacity=.5)
    shape=Rectangle(width=160,height=120,fill=FillStyle(paint=pattern))
    scene=Scene(160,120,background=Color(255,255,255));scene.add(shape)
    image=OpenCVRenderer().render(scene).buffer
    np.testing.assert_allclose(image[20,20],[127,127,255],atol=1)
    pattern.spacing=(8,8)
    pattern.transform=Transform(translation_x=8)
    shifted=render(shape)
    assert shifted[4,4,3]==0 and shifted[4,12,3] in (127,128)
