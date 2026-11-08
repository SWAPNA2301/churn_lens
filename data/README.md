# Dataset attribution

Dataset: **IBM Telco Customer Churn**, distributed with IBM's
[telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) sample project.

- [Upstream CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- [Upstream Apache License 2.0](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE)
- [Bundled license copy](../LICENSES/IBM-Apache-2.0.txt)

On 2026-11-08, the user-supplied `Telco-Customer-Churn.txt` and its
`data/raw/Telco-Customer-Churn.txt` copy were verified byte-for-byte identical
to the upstream CSV: **7,043 customer rows and 21 columns**. This verifies
content provenance, not the original download history of the supplied file.

SHA-256 for both local copies and the downloaded upstream file:
`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

The local files are renamed copies; their contents are unchanged. Cleaning
and transformations occur in the notebooks, not in these raw files.
IBM's repository distributes this sample under Apache-2.0. Preserve its
license and attribution when redistributing the dataset. The proposed MIT
license for ChurnLens code does not replace the third-party dataset license.
IBM is credited as the upstream publisher; no IBM endorsement is implied.
