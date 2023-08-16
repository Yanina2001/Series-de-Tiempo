import numpy as np
import exam_2

time_serie = np.array([ 
        1.8831507 , -1.34775906, -3.15734768, -2.50745871, -2.94513789,
       -1.29356462, -0.60601654, -1.49967919,  0.19950335,  2.62153882,
        4.96720129,  6.3103053 ,  5.27107142,  6.22341639,  6.28810385,
        6.32774818,  7.04345353,  6.65772406,  6.52852598,  7.10491974,
        7.23195663,  5.59757294,  4.78635343,  5.78322307,  5.17352585,
        4.1592548 ,  3.23335849,  4.5417096 ,  5.68446526,  5.85962795,
        4.70748   ,  4.39102315,  3.54721277,  2.88715572,  1.97850935
        ])

def test_covariance():
    length = time_serie.shape[0]
    
    t_covariances = np.array([10.307854356491486, 9.490662757888979, 8.075268807854785, 
                                6.645309088413593, 5.215517531352271, 3.751349160254289, 
                                2.1911812485521756, 0.7025562653950633, -0.5309348484409886, 
                                -1.542475644795888, -2.178911215045499, -2.6482588887871534, 
                                -2.9691125374578187, -3.181698437752862, -3.2824246183076955, 
                                -3.314903963389797, -3.1691980263812307, -2.7917114353116808, 
                                -2.4048377364510145, -2.1245625054144255, -1.9669085226022816, 
                                -1.911087037849526, -1.7806149363977637, -1.6086879528670242, 
                                -1.3502716621333017, -1.180420157090761, -1.067001977825851, 
                                -0.8817178839908252, -0.5443084668010546, -0.10967936315432425, 
                                0.22746688558256165, 0.3591861200378923, 0.39761109561631314, 
                                0.25432162784876583, 0.07537005120633876])
                                
    covariances = exam_2.covariance(time_serie, np.arange(length))

    assert np.allclose(covariances, t_covariances), "Unbiased covariances aren't correct"

    t_covariances = np.array([10.0133442320203, 9.219500964806436, 7.8445468419160775, 
                            6.4554431144589195, 5.066502744742206, 3.644167755675595, 
                            2.1285760700221132, 0.6824832292409186, -0.5157652813426746, 
                            -1.498404912087434, -2.116656608901342, -2.5725943491075207, 
                            -2.8842807506733097, -3.09079276810278, -3.1886410577846185, 
                            -3.22019242157866, -3.078649511341767, -2.711948251445633, 
                            -2.3361280868381282, -2.063860719545442, -1.9107111362422164, 
                            -1.8564845510538253, -1.7297402239292563, -1.5627254399279664, 
                            -1.311692471786636, -1.1466938668881679, -1.0365162070308267, 
                            -0.8565259444482302, -0.5287567963210245, -0.1065456670642007, 
                            0.22096783170877418, 0.34892365946538106, 0.3862507785987042, 
                            0.2470552956245154, 0.07321662117187193])

    covariances = exam_2.covariance(time_serie, np.arange(length), bias=True)

    assert np.allclose(covariances, t_covariances), "Biased covariances aren't correct"


def test_innovations():

    t_output= {
        "weights": np.array([
            [0.92072146, 0.        , 0.        ],
            [1.30962843, 0.78340928, 0.        ],
            [1.33534235, 1.24668277, 0.64468403]
        ]), 
        "variances": np.array([1.56959744, 1.28955543, 1.28477638])
    }

    output = exam_2.innovations(time_serie, 3)

    assert output["weights"].shape == t_output["weights"].shape, "Weights shape isn't correct"
    assert output["variances"].shape == t_output["variances"].shape, "Variances shape isn't correct"

    assert np.allclose(output["weights"], t_output["weights"]), "Weights values aren't correct"
    assert np.allclose(output["variances"], t_output["variances"]), "Variances values aren't correct"

def test_durbin_levinson():

    t_output= {
        "weights": np.array([
            [ 0.92072146,  0.        ,  0.        ],
            [ 1.30962843, -0.42239372,  0.        ],
            [ 1.33534235, -0.50211953,  0.06087667]
        ]), 
        "variances": np.array([1.56959744, 1.28955543, 1.28477638])
    }

    output = exam_2.durbin_levinson(time_serie, 3)

    assert output["weights"].shape == t_output["weights"].shape, "Weights shape isn't correct"
    assert output["variances"].shape == t_output["variances"].shape, "Variances shape isn't correct"

    assert np.allclose(output["weights"], t_output["weights"]), "Weights values aren't correct"
    assert np.allclose(output["variances"], t_output["variances"]), "Variances values aren't correct"

def test_ar():

   
    phis =  np.array([0.3470778764033191, 0.1605405904410016, -0.10078103794142566, -0.038806376816246754, -0.08659685210460862])
    thetas = np.array([0.9621657954103007, 0.5509350549042857, 0.5112479863826862, 0.5384396597681411])
    psis = np.array([1.3092436718136198, 1.1658851586528425, 1.0253066455407394, 0.9107194917412931, 0.8924847362324055, 0.8408748381050326, 0.5787665981336988, 0.5665500982131431, -0.1496408070438536])

    ar_roots = [(-1.6240936097286531+0j), (-0.5685986877496094+1.8817064197282432j), (-0.5685986877496094-1.8817064197282432j), (1.1565820545494176+0.7087977661713467j), (1.1565820545494176-0.7087977661713467j)]
    ma_roots = [(0.4607664962135094+1.1550875706561619j), (0.4607664962135094-1.1550875706561619j), (-0.9355160592405312+0.5707023276730591j), (-0.9355160592405312-0.5707023276730591j)]

    arma_model = exam_2.ARMA(p=5, q=4)   
    arma_model.train(time_serie)

    assert np.allclose(arma_model.phis, phis), "The computation of phis isn't correct"
    assert np.allclose(arma_model.thetas, thetas), "The computation of thetas isn't correct"
    assert np.allclose(arma_model.psis, psis), "The computation of psis isn't correct"

    assert np.allclose(arma_model.ar_roots, ar_roots), "The AR roots aren't correct"
    assert np.allclose(arma_model.ma_roots, ma_roots), "The MA roots aren't correct"

    assert arma_model.is_causal(), "The process should be causal"
    assert arma_model.is_invertible(), "The process should be invertible"





