import copy
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from drawcv import (Path, Point, Rectangle, Group, Color, FillStyle, VectorMask,
                    BoundingBox, Transform, Scene, OpenCVRenderer, Mask,
                    LinearGradient, GradientStop, ValidationError, BlurEffect, RenderError)
from drawcv.effects.clipping import ClipRect


def artwork(color=Color(255,255,255), opacity=1):
    return Rectangle(width=100,height=80,fill=FillStyle(color=color),opacity=opacity).to_path()


def render(node, alpha=True):
    scene=Scene(160,120,background=Color(0,0,0,0) if alpha else Color(255,255,255))
    scene.add(node)
    return OpenCVRenderer().render(scene,alpha=alpha).buffer


def host(mask):
    return Rectangle(width=150,height=110,fill=FillStyle(color=Color(255,0,0)),mask=mask)


@pytest.mark.parametrize('mode,color,expected',[
    ('alpha',Color(0,0,0),255),('alpha',Color(255,0,0),255),
    ('luminance',Color(0,0,0),0),('luminance',Color(255,255,255),255),
    ('luminance',Color(128,128,128),128),('luminance',Color(255,0,0),54),
    ('luminance',Color(0,255,0),182),('luminance',Color(0,0,255),18)])
def test_alpha_luminance_and_explicit_bounds(mode,color,expected):
    mask=VectorMask(artwork(color),BoundingBox(10,10,60,50),mode=mode)
    image=render(host(mask))
    assert image[30,30,3]==pytest.approx(expected,abs=1)
    assert image[30,80,3]==0 and image[5,5,3]==0


@pytest.mark.parametrize('mode',['alpha','luminance'])
@pytest.mark.parametrize('space',['object','world'])
def test_gradient_mask_matches_native_svg(mode,space):
    resvg=pytest.importorskip('resvg_py')
    colors=(Color(255,255,255,0),Color(255,255,255,1)) if mode=='alpha' else (Color(0,0,0),Color(255,255,255))
    art=artwork()
    art.fill=FillStyle(paint=LinearGradient(Point(0,0),Point(100,0),
                          (GradientStop(0,colors[0]),GradientStop(1,colors[1]))))
    mask=VectorMask(art,BoundingBox(5,5,90,70),mode=mode,space=space,
                    transform=Transform(translation_x=8),opacity=.8)
    node=host(mask)
    node.transform=Transform.from_matrix(np.array([[1.1,.2,5],[.1,1.,4],[0,0,1.]]))
    scene=Scene(160,120,background=Color(0,0,0,0));scene.add(node)
    actual=OpenCVRenderer().render(scene,alpha=True).buffer
    exported=scene.export_svg(strict=True)
    root=ET.fromstring(exported.svg)
    assert root.find('.//{http://www.w3.org/2000/svg}mask') is not None
    assert root.find('.//{http://www.w3.org/2000/svg}image') is None
    expected=cv2.imdecode(np.frombuffer(resvg.svg_to_bytes(svg_string=exported.svg),np.uint8),-1)
    for x,y in [(40,30),(60,40),(80,50)]:
        np.testing.assert_allclose(actual[y,x].astype(int),expected[y,x].astype(int),atol=3)


def test_group_layer_opacity_clip_and_bgr_compositing():
    mask=VectorMask(artwork(opacity=.5),BoundingBox(0,0,100,80),opacity=.5)
    node=host(mask);node.opacity=.5;node.clip=ClipRect(0,0,60,60)
    group=Group(children=[node],opacity=.5)
    image=render(group)
    assert image[30,30,3]==pytest.approx(255/16,abs=1)
    assert not image[:,65:,3].any()
    bgr=render(group,alpha=False)
    np.testing.assert_allclose(bgr[30,30],[239,239,255],atol=1)
    scene=Scene(160,120,background=Color(0,0,0,0));scene.add(artwork(Color(255,0,0)))
    scene.layers[0].mask=mask
    assert OpenCVRenderer().render(scene,alpha=True).buffer[30,30,3]==pytest.approx(64,abs=1)
    assert Scene.from_json(scene.to_json()).layers[0].mask.mode=='alpha'


def test_live_artwork_json_clone_state_and_source_parent_preserved():
    art=artwork();parent=Group(children=[art],transform=Transform(translation_x=100))
    mask=VectorMask(art,BoundingBox(0,0,100,80))
    node=host(mask)
    before=copy.deepcopy(art.to_dict())
    assert render(node)[30,30,3]==255
    assert art.parent is parent and art.to_dict()==before
    art.opacity=.25
    assert render(node)[30,30,3]==pytest.approx(64,abs=1)
    clone=node.clone();clone.mask.opacity=.5
    assert node.mask.opacity==1
    scene=Scene(160,120);scene.add(node)
    restored=Scene.from_json(scene.to_json()).objects[0]
    assert restored.mask.to_dict()==node.mask.to_dict()
    assert isinstance(Mask.from_dict(mask.to_dict()),VectorMask)
    state=node._get_semantic_state();node.mask=None;node._apply_semantic_state(state)
    assert isinstance(node.mask,VectorMask)


def test_mask_artwork_effects_and_svg_fallback():
    art=artwork();art.effects=[BlurEffect(kernel_size=5)]
    node=host(VectorMask(art,BoundingBox(0,0,100,80)))
    assert render(node)[30,30,3]==255
    scene=Scene(160,120);scene.add(node)
    assert scene.export_svg().fallbacks
    with pytest.raises(RenderError):
        scene.export_svg(strict=True)


def test_nested_masks_invalid_options_and_singular_transform():
    mask=VectorMask(artwork(),BoundingBox(0,0,100,80))
    with pytest.raises(ValidationError):
        mask.mode='brightness'
    assert mask.mode=='alpha'
    with pytest.raises(ValidationError):
        VectorMask(artwork(),BoundingBox(0,0,0,80))
    mask.transform=Transform.from_matrix(np.diag([0.,1.,1.]))
    assert not render(host(mask))[...,3].any()
    mask.artwork.mask=mask
    with pytest.raises(ValidationError,match='Nested'):
        mask.to_dict()
    with pytest.raises(ValidationError,match='Nested'):
        render(host(mask))
