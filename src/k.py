
    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd_reg1_06_05_impmeancurv.json'),'w') as fwrite:
        json.dump(svd_reg1_impmeancurv, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_logdet_reg1_06_05_impmeancurv.json'),'w') as fwrite:
        json.dump(logdet_reg1_impmeancurv, fwrite)

    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd_reg1_06_05_impmediancurv.json'),'w') as fwrite:
        json.dump(svd_reg1_impmediancurv, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_logdet_reg1_06_05_impmediancurv.json'),'w') as fwrite:
        json.dump(logdet_reg1_impmediancurv, fwrite)

    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd_reg1_06_05_impthresh.json'),'w') as fwrite:
        json.dump(svd_reg1_impthresh, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_logdet_reg1_06_05_impthresh.json'),'w') as fwrite:
        json.dump(logdet_reg1_impthresh, fwrite)

    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd_reg1_06_05_chamdist.json'),'w') as fwrite:
        json.dump(svd_reg1_chamdist, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_logdet_reg1_06_05_chamdist.json'),'w') as fwrite:
        json.dump(logdet_reg1_chamdist, fwrite)

    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd_reg1_06_05_discmeancurv.json'),'w') as fwrite:
        json.dump(svd_reg1_discmeancurv, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_logdet_reg1_06_05_discmeancurv.json'),'w') as fwrite:
        json.dump(logdet_reg1_discmeancurv, fwrite)

    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd_reg1_06_05_discmediancurv.json'),'w') as fwrite:
        json.dump(svd_reg1_discmediancurv, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_logdet_reg1_06_05_discmediancurv.json'),'w') as fwrite:
        json.dump(logdet_reg1_discmediancurv, fwrite)

    with open(os.path.join(args.testoutputdir, args.testfilename+'_svd_reg1_06_05_discthresh.json'),'w') as fwrite:
        json.dump(svd_reg1_discthresh, fwrite)
    with open(os.path.join(args.testoutputdir, args.testfilename+'_logdet_reg1_06_05_discthresh.json'),'w') as fwrite:
        json.dump(logdet_reg1_discthresh, fwrite)

    avg_svd_reg1_impmedian = np.array(list(svd_reg1_impmediancurv.values())).mean()
    avg_logdet_reg1_impmedian = np.array(list(logdet_reg1_impmediancurv.values())).mean()

    avg_svd_reg1_impmean = np.array(list(svd_reg1_impmeancurv.values())).mean()
    avg_logdet_reg1_impmean = np.array(list(logdet_reg1_impmeancurv.values())).mean()

    avg_svd_reg1_impthresh = 100*np.array(list(svd_reg1_impthresh.values())).mean()
    avg_logdet_reg1_impthresh = 100*np.array(list(logdet_reg1_impthresh.values())).mean()

    avg_svd_reg1_discmedian = np.array(list(svd_reg1_discmediancurv.values())).mean()
    avg_logdet_reg1_discmedian = np.array(list(logdet_reg1_discmediancurv.values())).mean()

    avg_svd_reg1_discmean = np.array(list(svd_reg1_discmeancurv.values())).mean()
    avg_logdet_reg1_discmean = np.array(list(logdet_reg1_discmeancurv.values())).mean()

