from brukerview.jcampdx import read_jcampdx

SAMPLE = """##TITLE=Parameter List
##JCAMPDX=4.24
$$ a comment line
##$VisuCoreFrameCount=29
##$VisuCoreDim=2
##$VisuCoreSize=( 2 )
192 192
##$VisuCoreExtent=( 2 )
24 24.5
##$VisuCoreUnits=( 2, 65 )
<mm> <mm>
##$VisuCoreDataSlope=( 4 )
@3*(0.5) 2
##$VisuCoreWordType=_16BIT_SGN_INT
##$VisuFGOrderDesc=( 2 )
(32, <FG_ECHO>, <>, 0, 1) (7, <FG_SLICE>, <>, 1, 2)
##$VisuCoreSlicePacksDef=(1, 1)
##$VisuAcquisitionProtocol=( 65 )
<T2_TurboRARE>
##$VisuCoreOrientation=( 2, 9 )
1 0 0 0 1 0 0 0 1 1 0 0 0 1 0
0 0 1
##$VisuSubjectWeight=0.03
##END=
"""


def test_parse(tmp_path):
    p = tmp_path / 'visu_pars'
    p.write_text(SAMPLE)
    d = read_jcampdx(p)
    assert d['VisuCoreFrameCount'] == 29
    assert d['VisuCoreSize'] == [192, 192]
    assert d['VisuCoreExtent'] == [24, 24.5]
    assert d['VisuCoreUnits'] == ['mm', 'mm']
    assert d['VisuCoreDataSlope'] == [0.5, 0.5, 0.5, 2]
    assert d['VisuCoreWordType'] == '_16BIT_SGN_INT'
    assert d['VisuFGOrderDesc'] == [(32, 'FG_ECHO', '', 0, 1), (7, 'FG_SLICE', '', 1, 2)]
    assert d['VisuCoreSlicePacksDef'] == (1, 1)
    assert d['VisuAcquisitionProtocol'] == 'T2_TurboRARE'
    assert d['VisuCoreOrientation'] == [[1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1]]
    assert d['VisuSubjectWeight'] == 0.03
    assert d['TITLE'] == 'Parameter List'
